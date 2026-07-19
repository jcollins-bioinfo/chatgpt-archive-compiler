"""Synthetic tests for persistent API cost authorization and budget planning."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from chatgpt_archive_compiler.exceptions import ApiBudgetExceededError
from chatgpt_archive_compiler.semantic.budget import (
    ApiBudget,
    ModelTokenPrice,
    estimate_budgeted_semantic_cost,
)
from chatgpt_archive_compiler.semantic.models import SemanticRunEstimate


def test_budget_reserves_before_calls_settles_usage_and_resumes(tmp_path: Path) -> None:
    """Durable charged cost falls to reported usage and survives a new ledger instance."""

    path = tmp_path / "api_budget.json"
    price = ModelTokenPrice(input_usd_per_million=1, output_usd_per_million=6)
    budget = ApiBudget(max_cost_usd="1.00", ledger_path=path)
    request_id = budget.reserve(
        stage="synthetic",
        model="synthetic-model",
        price=price,
        estimated_input_tokens=10_000,
        max_output_tokens=20_000,
    )
    reserved = budget.snapshot()
    budget.settle(
        request_id,
        actual_input_tokens=8_000,
        actual_output_tokens=2_000,
    )
    settled = budget.snapshot()
    resumed = ApiBudget(max_cost_usd="1.00", ledger_path=path).snapshot()

    assert reserved.charged_cost_usd > settled.charged_cost_usd
    assert settled.charged_cost_usd == Decimal("0.020")
    assert resumed == settled
    assert resumed.completed_request_count == 1


def test_budget_rejects_request_before_transmission_and_retains_failed_reserve(
    tmp_path: Path,
) -> None:
    """Worst-case output authorization is enforced and uncertain calls remain charged."""

    budget = ApiBudget(max_cost_usd="0.10", ledger_path=tmp_path / "ledger.json")
    price = ModelTokenPrice(input_usd_per_million=1, output_usd_per_million=6)

    with pytest.raises(ApiBudgetExceededError, match="before transmission"):
        budget.reserve(
            stage="too-large",
            model="synthetic-model",
            price=price,
            estimated_input_tokens=1_000,
            max_output_tokens=100_000,
        )
    assert budget.snapshot().request_count == 0

    request_id = budget.reserve(
        stage="uncertain",
        model="synthetic-model",
        price=price,
        estimated_input_tokens=1_000,
        max_output_tokens=10_000,
    )
    reserved_cost = budget.snapshot().charged_cost_usd
    budget.fail(request_id)
    assert budget.snapshot().charged_cost_usd == reserved_cost
    assert budget.snapshot().failed_or_unsettled_request_count == 1


def test_budgeted_plan_limits_model_profiles_to_category_representatives() -> None:
    """The planner prices complete local coverage but only bounded external refinement."""

    estimate = SemanticRunEstimate(
        conversation_count=5_000,
        message_count=25_000,
        character_count=80_000_000,
        original_character_count=80_000_000,
        approximate_input_tokens=20_000_000,
        embedding_item_count=5_000,
        analysis_item_count=5_000,
        truncated_conversation_count=0,
    )
    embedding_price = ModelTokenPrice(input_usd_per_million="0.02")
    luna = ModelTokenPrice(input_usd_per_million="1.00", output_usd_per_million="6.00")
    terra = ModelTokenPrice(input_usd_per_million="2.50", output_usd_per_million="15.00")

    plan = estimate_budgeted_semantic_cost(
        estimate,
        embedding_price=embedding_price,
        profile_price=luna,
        taxonomy_price=luna,
        synthesis_price=terra,
        hard_budget_usd="5.00",
        max_leaf_categories=48,
        refined_conversations_per_category=3,
        max_refined_conversations=144,
        analysis_batch_size=8,
        embedding_max_tokens_per_item=4_000,
        profile_max_input_tokens_per_item=4_000,
        profile_max_output_tokens_per_batch=6_000,
        taxonomy_max_output_tokens=16_000,
        synthesis_max_output_tokens=24_000,
    )

    assert plan.locally_profiled_conversation_count == 5_000
    assert plan.model_refined_conversation_count == 144
    assert plan.profile_request_count == 18
    assert plan.expected_cost_usd < Decimal("3.00")
    assert plan.conservative_scheduled_reserve_usd < plan.hard_budget_usd
    assert plan.scheduled_plan_fits_hard_budget
