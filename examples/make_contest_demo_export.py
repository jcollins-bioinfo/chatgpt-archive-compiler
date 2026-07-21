"""Generate the deterministic, fictional 72-conversation contest demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path
from typing import Any

START = 1_751_328_000
DOMAINS = {
    "helix": ("HelixFold research", "computational biology", "validate the held-out cohort"),
    "atlas": ("Lumen Atlas software", "privacy-first software", "choose a local deployment model"),
    "music": ("Cadence Loom", "music analysis", "finish audio-to-score alignment"),
    "energy": ("Northstar apartment", "energy efficiency", "compare winter heat-pump data"),
    "career": ("Research engineering search", "career planning", "send the portfolio follow-up"),
    "philosophy": ("Long-form inquiry", "philosophy of explanation", "resolve the realism outline"),
}
STAGES = (
    "initial hypothesis",
    "literature map",
    "prototype",
    "failed test",
    "reframing",
    "design decision",
    "implementation",
    "review",
    "milestone",
    "dormant return",
    "release planning",
    "open question",
)


def _conversation(
    index: int, domain: str, stage: str, timestamp: int, rng: random.Random
) -> dict[str, Any]:
    project, theme, open_question = DOMAINS[domain]
    prefix = f"contest-{index:03d}"
    root, user, assistant = f"{prefix}-root", f"{prefix}-user", f"{prefix}-assistant"
    user_text = (
        f"For {project}, revisit the {stage}. Connect it to our earlier {theme} decisions "
        "and preserve evidence."
    )
    if stage == "open question":
        user_text += f" We still need to {open_question}; leave this unresolved."
    assistant_text = (
        f"Candidate {stage} for {project}: document inputs, decision, result, and source links. "
        "This is a synthetic record, not a factual claim."
    )
    mapping: dict[str, Any] = {
        root: {"id": root, "parent": None, "children": [user], "message": None},
        user: {
            "id": user,
            "parent": root,
            "children": [assistant],
            "message": {
                "id": f"{user}-message",
                "author": {"role": "user"},
                "create_time": timestamp,
                "content": {"content_type": "text", "parts": [user_text]},
                "metadata": {},
            },
        },
        assistant: {
            "id": assistant,
            "parent": user,
            "children": [],
            "message": {
                "id": f"{assistant}-message",
                "author": {"role": "assistant"},
                "create_time": timestamp + 90,
                "content": {"content_type": "text", "parts": [assistant_text]},
                "metadata": {"model_slug": "synthetic-local-demo"},
            },
        },
    }
    if index % 17 == 0:
        alternate = f"{prefix}-assistant-alternate"
        mapping[user]["children"].append(alternate)
        mapping[alternate] = {
            "id": alternate,
            "parent": user,
            "children": [],
            "message": {
                "id": f"{alternate}-message",
                "author": {"role": "assistant"},
                "create_time": timestamp + 60,
                "content": {
                    "content_type": "text",
                    "parts": [
                        f"Alternate synthetic framing for {project}, preserved but not selected."
                    ],
                },
                "metadata": {},
            },
        }
    return {
        "id": f"{prefix}-conversation",
        "title": f"{project} · {stage}",
        "create_time": timestamp,
        "update_time": timestamp + rng.randint(100, 900),
        "current_node": assistant,
        "mapping": mapping,
    }


def payload(seed: int) -> list[dict[str, Any]]:
    """Return 72 chronological records plus professional-safety test cases."""

    rng = random.Random(seed)
    records = [
        _conversation(index + 1, domain, STAGES[round_index], START + index * 388_800, rng)
        for index, (round_index, domain) in enumerate(
            stage_domain
            for stage_domain in (
                (round_index, domain) for round_index in range(12) for domain in DOMAINS
            )
        )
    ]
    sensitive = {
        5: "Mark this private: my partner and I discussed an intimate breakup.",
        11: "My diagnosis and my medication are private mental health information.",
        23: "I cannot pay rent and my debt is causing financial hardship.",
        35: "My manager raised a workplace grievance that needs contextual review.",
        47: "Synthetic credential API_KEY=fictional_demo_secret_12345 must never be shared.",
        59: "The database uses a medication table as a purely technical schema term.",
    }
    for index, text in sensitive.items():
        user_node = records[index]["mapping"][f"contest-{index + 1:03d}-user"]
        user_node["message"]["content"]["parts"] = [text]
    return records


def ground_truth(seed: int) -> dict[str, Any]:
    """Return evaluation-only planted structure, never included in the export ZIP."""

    projects = []
    for offset, (key, (title, theme, open_loop)) in enumerate(DOMAINS.items()):
        ids = [
            f"contest-{offset + 1 + 6 * round_index:03d}-conversation" for round_index in range(12)
        ]
        projects.append(
            {
                "id": key,
                "title": title,
                "theme": theme,
                "conversation_ids": ids,
                "milestone_dates": [START + (offset + 6 * stage) * 388_800 for stage in (0, 4, 8)],
                "open_loop": open_loop,
            }
        )
    return {
        "schema_version": "1.0",
        "seed": seed,
        "fictional": True,
        "projects": projects,
        "known_resolved_loops": ["Northstar draft sealing completed"],
        "preference_changes": ["Lumen Atlas changed from hosted-first to local-first"],
        "contradictions": [
            {"kind": "true-candidate", "summary": "Two incompatible release dates"},
            {
                "kind": "context-change",
                "summary": "Heating choice differs between rental and owned-home scenarios",
            },
        ],
        "cross_conversation_relationships": [
            ["contest-001-conversation", "contest-037-conversation"]
        ],
        "professional_safety": {
            "exclude_indexes": [5, 11, 23, 47],
            "review_indexes": [35],
            "false_positive_trap": 59,
        },
        "user_correction_example": {
            "type": "rename",
            "object_id": "atlas",
            "new_value": "Lumen Atlas",
        },
    }


def write_demo(output: Path, truth_path: Path, seed: int) -> str:
    """Write stable ZIP bytes and external ground truth; return SHA-256 fingerprint."""

    conversations = json.dumps(
        payload(seed), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    readme = (
        b"FICTIONAL SYNTHETIC DATA ONLY. Generated offline for the ChatGPT Archive Compiler "
        b"contest demo.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in (("conversations.json", conversations), ("README_SYNTHETIC.txt", readme)):
            member = zipfile.ZipInfo(name, date_time=(2026, 7, 21, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, data)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(
        json.dumps(ground_truth(seed), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    print(f"{write_demo(args.output, args.ground_truth, args.seed)}  {args.output}")


if __name__ == "__main__":
    main()
