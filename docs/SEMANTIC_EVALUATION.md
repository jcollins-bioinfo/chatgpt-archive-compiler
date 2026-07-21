# Semantic evaluation

## Tasks and ground truth

The offline generator creates 102 naturally varied, multi-turn fictional conversations across six
sustained projects and fourteen months, then writes a separate ground-truth JSON file. Planted arcs
include failed controls, methodological reversals, dormant returns, open questions, alternate
branches, preference changes, contextual contradictions, cross-domain links, and
professional-safety cases. Ground truth never enters the export or inference pipeline.

## Metrics

Evaluation should report separate metrics—not a single definitive score: project assignment
precision/recall after maximum-overlap project matching; token-set/Jaccard theme recovery;
milestone date absolute error; open-loop precision/recall; evidence keys resolving to catalog
records; contradiction-candidate precision by planted type; supported semantic-object coverage;
unsupported claims; and normalized duplicate-insight rate. Label matching uses conversation-set
overlap first and normalized label similarity only as a tie-breaker.

## Current baseline and limitations

The deterministic local pipeline provides reproducible hashing representations, graph communities,
heuristic profiles, taxonomy, timelines, review signals, and referential-integrity safeguards. Its
semantic usefulness has **not** been validated against a blinded human rubric. This change validates
coarse corpus recovery, safety-boundary placement, and evidence referential integrity in automated
tests; it does not claim psychological insight, calibrated confidence, contradiction adjudication,
or causal inference.

Known failure modes include generic category names, sparse text fragmenting a project, vocabulary
overlap joining unrelated threads, missed implicit open loops, and conservative privacy rules.
Next experiments are explicit benchmark scoring, reviewer agreement, alias-aware project matching,
and bounded opt-in structured model refinement.

Run the acceptance harness through the real local semantic pipeline:

```bash
uv run python examples/validate_contest_demo.py \
  --archive /tmp/chatgpt-archive-contest-demo.zip \
  --ground-truth /tmp/chatgpt-archive-contest-ground-truth.json \
  --output-directory /tmp/chatgpt-archive-semantic-validation \
  --report /tmp/chatgpt-archive-demo-validation.json
```

It fails unless project timelines, overlap matching, longitudinal findings, open loops, dormant
threads, reversals, cross-domain connections, category-name uniqueness, generic-term safeguards,
and evidence referential integrity all meet their independent thresholds. Its JSON diagnostics name
missed, fragmented, or merged projects; missed milestones; unsupported findings; duplicate labels;
generic terms; and evidence-link failures.
