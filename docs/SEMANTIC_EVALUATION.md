# Semantic evaluation

## Tasks and ground truth

The offline generator creates 72 fictional conversations across six recurring domains and writes a
separate ground-truth JSON file. Planted arcs include stages, dormant returns, open questions,
alternate branches, context-dependent preference changes, contradiction candidates, cross-domain
links, and professional-safety cases. Ground truth never enters the export or inference pipeline.

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
