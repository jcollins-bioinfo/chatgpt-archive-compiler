"""Persistent, pre-request API cost authorization for semantic analysis.

The ledger is deliberately independent of the OpenAI SDK. Provider adapters reserve the maximum
configured cost of each request before sending it, then replace that reservation with reported
usage when a response arrives. An interrupted or unobservable request retains its conservative
reservation so restarting a Colab runtime cannot silently reset the spending ceiling.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chatgpt_archive_compiler.exceptions import ApiBudgetExceededError
from chatgpt_archive_compiler.semantic.models import SemanticRunEstimate

_MILLION = Decimal(1_000_000)
_CENT = Decimal("0.01")


class ModelTokenPrice(BaseModel):
    """Token prices for one model, expressed in US dollars per million tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    input_usd_per_million: Decimal = Field(ge=0)
    output_usd_per_million: Decimal = Field(default=Decimal(0), ge=0)

    def cost(self, input_tokens: int, output_tokens: int = 0) -> Decimal:
        """Return the exact configured cost for non-negative token counts."""

        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must not be negative")
        return (
            Decimal(input_tokens) * self.input_usd_per_million
            + Decimal(output_tokens) * self.output_usd_per_million
        ) / _MILLION


class ApiBudgetEntry(BaseModel):
    """One content-free API request reservation and its eventual accounting state."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    request_id: str
    created_at: datetime
    stage: str
    model: str
    price: ModelTokenPrice
    reserved_input_tokens: int = Field(ge=0)
    reserved_output_tokens: int = Field(ge=0)
    actual_input_tokens: int | None = Field(default=None, ge=0)
    actual_output_tokens: int | None = Field(default=None, ge=0)
    reserved_cost_usd: Decimal = Field(ge=0)
    charged_cost_usd: Decimal = Field(ge=0)
    status: Literal["reserved", "completed", "failed"]


class ApiBudgetState(BaseModel):
    """Durable private ledger state loaded across interrupted notebook sessions."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    ledger_version: str = "1.0"
    max_cost_usd: Decimal = Field(gt=0)
    entries: tuple[ApiBudgetEntry, ...] = ()


class ApiBudgetSnapshot(BaseModel):
    """Content-free budget summary suitable for notebook display and manifests."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_cost_usd: Decimal
    charged_cost_usd: Decimal
    remaining_cost_usd: Decimal
    request_count: int = Field(ge=0)
    completed_request_count: int = Field(ge=0)
    failed_or_unsettled_request_count: int = Field(ge=0)


class BudgetedSemanticCostPlan(BaseModel):
    """Content-free expected and conservative costs for a selective semantic run."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    conversation_count: int = Field(ge=0)
    locally_profiled_conversation_count: int = Field(ge=0)
    model_refined_conversation_count: int = Field(ge=0)
    maximum_leaf_category_count: int = Field(gt=0)
    profile_request_count: int = Field(ge=0)
    estimated_embedding_input_tokens: int = Field(ge=0)
    estimated_structured_input_tokens: int = Field(ge=0)
    estimated_structured_output_tokens: int = Field(ge=0)
    expected_cost_usd: Decimal = Field(ge=0)
    conservative_scheduled_reserve_usd: Decimal = Field(ge=0)
    hard_budget_usd: Decimal = Field(gt=0)
    scheduled_plan_fits_hard_budget: bool


def estimate_budgeted_semantic_cost(
    estimate: SemanticRunEstimate,
    *,
    embedding_price: ModelTokenPrice,
    profile_price: ModelTokenPrice,
    taxonomy_price: ModelTokenPrice,
    synthesis_price: ModelTokenPrice,
    hard_budget_usd: Decimal | float | str,
    max_leaf_categories: int,
    refined_conversations_per_category: int,
    max_refined_conversations: int,
    analysis_batch_size: int,
    embedding_max_tokens_per_item: int,
    profile_max_input_tokens_per_item: int,
    profile_max_output_tokens_per_batch: int,
    taxonomy_max_output_tokens: int,
    synthesis_max_output_tokens: int,
) -> BudgetedSemanticCostPlan:
    """Estimate a hierarchical run using only counts and explicit bounded-work assumptions."""

    positive = (
        max_leaf_categories,
        refined_conversations_per_category,
        analysis_batch_size,
        embedding_max_tokens_per_item,
        profile_max_input_tokens_per_item,
        profile_max_output_tokens_per_batch,
        taxonomy_max_output_tokens,
        synthesis_max_output_tokens,
    )
    if min(positive) <= 0 or max_refined_conversations < 0:
        raise ValueError("budgeted cost-planning limits must be positive or explicitly zero")
    maximum = Decimal(str(hard_budget_usd))
    if maximum <= 0:
        raise ValueError("hard_budget_usd must be positive")

    conversation_count = estimate.conversation_count
    refined_count = min(
        conversation_count,
        max_refined_conversations,
        max_leaf_categories * refined_conversations_per_category,
    )
    profile_request_count = (
        (refined_count + analysis_batch_size - 1) // analysis_batch_size if refined_count else 0
    )
    embedding_tokens = min(
        estimate.approximate_input_tokens,
        conversation_count * embedding_max_tokens_per_item,
    )
    average_tokens = (
        (estimate.approximate_input_tokens + conversation_count - 1) // conversation_count
        if conversation_count
        else 0
    )
    profile_input_tokens = refined_count * min(average_tokens, profile_max_input_tokens_per_item)
    taxonomy_input_tokens = 2_000 + max_leaf_categories * 700 + refined_count * 160
    synthesis_input_tokens = 3_000 + max_leaf_categories * 1_100 + refined_count * 220
    structured_input_tokens = profile_input_tokens + taxonomy_input_tokens + synthesis_input_tokens
    expected_profile_output = refined_count * 180
    expected_taxonomy_output = max_leaf_categories * 150
    expected_synthesis_output = 8_000
    structured_output_tokens = (
        expected_profile_output + expected_taxonomy_output + expected_synthesis_output
    )
    expected_cost = (
        embedding_price.cost(embedding_tokens)
        + profile_price.cost(profile_input_tokens, expected_profile_output)
        + taxonomy_price.cost(taxonomy_input_tokens, expected_taxonomy_output)
        + synthesis_price.cost(synthesis_input_tokens, expected_synthesis_output)
    )

    def reserved_input(tokens: int, requests: int = 1) -> int:
        return math.ceil(tokens * 1.20) + requests * 512

    conservative_reserve = (
        embedding_price.cost(math.ceil(embedding_tokens * 1.20))
        + profile_price.cost(
            reserved_input(profile_input_tokens, profile_request_count),
            profile_request_count * profile_max_output_tokens_per_batch,
        )
        + taxonomy_price.cost(reserved_input(taxonomy_input_tokens), taxonomy_max_output_tokens)
        + synthesis_price.cost(reserved_input(synthesis_input_tokens), synthesis_max_output_tokens)
    )
    return BudgetedSemanticCostPlan(
        conversation_count=conversation_count,
        locally_profiled_conversation_count=conversation_count,
        model_refined_conversation_count=refined_count,
        maximum_leaf_category_count=max_leaf_categories,
        profile_request_count=profile_request_count,
        estimated_embedding_input_tokens=embedding_tokens,
        estimated_structured_input_tokens=structured_input_tokens,
        estimated_structured_output_tokens=structured_output_tokens,
        expected_cost_usd=expected_cost,
        conservative_scheduled_reserve_usd=conservative_reserve,
        hard_budget_usd=maximum,
        scheduled_plan_fits_hard_budget=conservative_reserve <= maximum,
    )


class ApiBudget:
    """Authorize sequential API requests against a persistent hard spending ceiling.

    Parameters
    ----------
    max_cost_usd
        Maximum cumulative configured cost for this ledger. A pre-existing ledger must use the
        same ceiling, preventing a resumed notebook from silently changing its authorization.
    ledger_path
        Optional JSON destination. Reservations are atomically persisted before network calls.
    input_safety_factor
        Multiplier applied to estimated input tokens before authorization. Output reservations use
        the request's exact ``max_output_tokens`` value and therefore need no multiplier.
    """

    def __init__(
        self,
        *,
        max_cost_usd: Decimal | float | str,
        ledger_path: str | Path | None = None,
        input_safety_factor: float = 1.20,
    ) -> None:
        maximum = Decimal(str(max_cost_usd))
        if maximum <= 0:
            raise ValueError("max_cost_usd must be positive")
        if not math.isfinite(input_safety_factor) or input_safety_factor < 1.0:
            raise ValueError("input_safety_factor must be finite and at least 1.0")
        self._ledger_path = Path(ledger_path).expanduser().resolve() if ledger_path else None
        self._input_safety_factor = input_safety_factor
        self._state = self._load_or_initialize(maximum)

    @property
    def max_cost_usd(self) -> Decimal:
        """Return the immutable cumulative spending ceiling."""

        return self._state.max_cost_usd

    @property
    def ledger_path(self) -> Path | None:
        """Return the durable ledger path when persistence is enabled."""

        return self._ledger_path

    def snapshot(self) -> ApiBudgetSnapshot:
        """Return aggregate authorization state without source-derived content."""

        charged = sum((entry.charged_cost_usd for entry in self._state.entries), Decimal(0))
        remaining = max(Decimal(0), self.max_cost_usd - charged)
        completed = sum(entry.status == "completed" for entry in self._state.entries)
        unsettled = len(self._state.entries) - completed
        return ApiBudgetSnapshot(
            max_cost_usd=self.max_cost_usd,
            charged_cost_usd=charged,
            remaining_cost_usd=remaining,
            request_count=len(self._state.entries),
            completed_request_count=completed,
            failed_or_unsettled_request_count=unsettled,
        )

    def reserve(
        self,
        *,
        stage: str,
        model: str,
        price: ModelTokenPrice,
        estimated_input_tokens: int,
        max_output_tokens: int = 0,
        fixed_input_token_allowance: int = 0,
    ) -> str:
        """Persist a worst-case request reservation or reject it before any network call."""

        if estimated_input_tokens < 0 or max_output_tokens < 0 or fixed_input_token_allowance < 0:
            raise ValueError("budget token counts must not be negative")
        reserved_input = (
            math.ceil(estimated_input_tokens * self._input_safety_factor)
            + fixed_input_token_allowance
        )
        reserved_cost = price.cost(reserved_input, max_output_tokens)
        snapshot = self.snapshot()
        if snapshot.charged_cost_usd + reserved_cost > self.max_cost_usd:
            required = reserved_cost.quantize(_CENT)
            remaining = snapshot.remaining_cost_usd.quantize(_CENT)
            raise ApiBudgetExceededError(
                f"API budget stopped stage {stage!r} before transmission: request reserve "
                f"${required} exceeds the remaining ${remaining}."
            )
        entry = ApiBudgetEntry(
            request_id=uuid.uuid4().hex,
            created_at=datetime.now(UTC),
            stage=stage,
            model=model,
            price=price,
            reserved_input_tokens=reserved_input,
            reserved_output_tokens=max_output_tokens,
            reserved_cost_usd=reserved_cost,
            charged_cost_usd=reserved_cost,
            status="reserved",
        )
        self._state = self._state.model_copy(update={"entries": (*self._state.entries, entry)})
        self._persist()
        return entry.request_id

    def settle(
        self,
        request_id: str,
        *,
        actual_input_tokens: int | None,
        actual_output_tokens: int | None = None,
    ) -> ApiBudgetEntry:
        """Replace a reservation with provider-reported usage when it is available."""

        index, entry = self._entry(request_id)
        if entry.status != "reserved":
            raise ValueError("only a reserved API request can be settled")
        if actual_input_tokens is not None and actual_input_tokens < 0:
            raise ValueError("actual_input_tokens must not be negative")
        if actual_output_tokens is not None and actual_output_tokens < 0:
            raise ValueError("actual_output_tokens must not be negative")
        if actual_input_tokens is None or (
            entry.price.output_usd_per_million > 0 and actual_output_tokens is None
        ):
            charged = entry.reserved_cost_usd
        else:
            charged = entry.price.cost(actual_input_tokens, actual_output_tokens or 0)
        completed = entry.model_copy(
            update={
                "actual_input_tokens": actual_input_tokens,
                "actual_output_tokens": actual_output_tokens,
                "charged_cost_usd": charged,
                "status": "completed",
            }
        )
        self._replace(index, completed)
        if self.snapshot().charged_cost_usd > self.max_cost_usd:
            raise ApiBudgetExceededError(
                "Provider-reported usage exceeded its conservative reservation; "
                "the API budget has stopped all further requests."
            )
        return completed

    def fail(self, request_id: str) -> ApiBudgetEntry:
        """Conservatively retain a request's reservation when success or billing is unknown."""

        index, entry = self._entry(request_id)
        if entry.status != "reserved":
            return entry
        failed = entry.model_copy(update={"status": "failed"})
        self._replace(index, failed)
        return failed

    def _load_or_initialize(self, maximum: Decimal) -> ApiBudgetState:
        """Load compatible durable state or create an empty ledger."""

        if self._ledger_path is None or not self._ledger_path.is_file():
            state = ApiBudgetState(max_cost_usd=maximum)
            self._state = state
            self._persist()
            return state
        try:
            state = ApiBudgetState.model_validate_json(
                self._ledger_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError, ValueError) as exc:
            raise ValueError("Existing API budget ledger is unreadable or invalid.") from exc
        if state.max_cost_usd != maximum:
            raise ValueError("Existing API budget ledger uses a different spending ceiling.")
        return state

    def _entry(self, request_id: str) -> tuple[int, ApiBudgetEntry]:
        """Resolve a request ID without exposing any provider payload."""

        for index, entry in enumerate(self._state.entries):
            if entry.request_id == request_id:
                return index, entry
        raise KeyError("Unknown API budget reservation.")

    def _replace(self, index: int, entry: ApiBudgetEntry) -> None:
        """Replace one immutable ledger entry and persist the new state."""

        entries = list(self._state.entries)
        entries[index] = entry
        self._state = self._state.model_copy(update={"entries": tuple(entries)})
        self._persist()

    def _persist(self) -> None:
        """Atomically persist the complete content-free ledger when configured."""

        if self._ledger_path is None:
            return
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self._ledger_path.parent,
                prefix=f".{self._ledger_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(
                    self._state.model_dump(mode="json"),
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._ledger_path)
            temporary = None
        except OSError as exc:
            raise ValueError("Could not persist the API budget ledger.") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
