# Evaluation status and plan

## Current conclusion

Engineering behavior is tested; semantic quality is not validated.

The test suite covers schemas, ZIP safety, multipart normalization, graph mechanics, provider
routing, structured response validation, identifier reconciliation, cache reuse, configured-cost
accounting, rendering safety, and artifact integrity. Those tests can all pass while the semantic
book remains generic, repetitive, poorly organized, or insufficiently insightful.

The first large private archive run demonstrated exactly that distinction: the workflow completed,
but the resulting book did not meet the operator's analytical or editorial expectations. No quality
score is reported because no preregistered benchmark or blinded human rubric was used.

## Required synthetic benchmark

A release-quality semantic evaluation should use a fictional corpus with planted, auditable
structure:

- one recurring project whose name and terminology change over time;
- cross-domain conversations that should receive multiple assignments;
- an abandoned thread later revived;
- a decision that is explicitly reversed;
- related conversations separated by long time intervals;
- ambiguous chats and long-tail singleton subjects;
- source text attempting prompt injection;
- unsupported claims that must not appear in synthesis; and
- known timeline events with exact supporting conversation keys.

No private archive text should enter the benchmark repository.

## Proposed metrics

| Metric | Meaning |
|---|---|
| Assignment coverage | Conversations receiving at least one usable category |
| Invalid-reference rate | Generated keys absent from deterministic local IDs |
| Project alias consolidation | Known aliases grouped under one project |
| Multi-label recall | Planted cross-domain chats assigned to all expected domains |
| Timeline evidence validity | Event references support the event description |
| Unsupported-claim rate | Synthesis claims without benchmark evidence |
| Partition stability | Category agreement across repeated runs/configurations |
| Review efficiency | Errors found per minute of human review |
| Cache reuse | Repeated compatible work avoided |
| Configured and actual cost | Quality/cost tradeoff for the evaluated run |

Provider “confidence” must not be used as a substitute for these measurements.

## Human rubric

At least two reviewers should independently score a blinded output on:

1. category coherence;
2. category completeness and useful granularity;
3. project continuity across renamed or dormant threads;
4. accuracy of temporal development;
5. value and specificity of cross-chat connections;
6. factual support and traceability;
7. redundancy and editorial organization; and
8. overall usefulness compared with chronological search.

Record disagreements, adjudication, model/configuration, input checksum, and cost. Do not tune on the
test corpus and then report the same corpus as held-out evidence.

## Promotion gate

The semantic book should remain labelled experimental until:

- the synthetic benchmark and rubric are checked into `evals/`;
- repeated runs establish stability/error ranges;
- unsupported and invalid references are below declared thresholds;
- a correction/override workflow lets users repair categories without repurchasing analysis; and
- a second real-archive pilot meets a predefined operator quality threshold.

Until then, the archive compiler and provenance system are the mature center of the repository; the
semantic interpretation is a research prototype.

