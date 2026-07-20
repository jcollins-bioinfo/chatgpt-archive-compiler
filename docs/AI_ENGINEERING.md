# AI engineering system card

## Purpose and status

The semantic atlas is an experimental interpretation layer over a loss-aware archive compiler. Its
engineering goal is to make model-assisted analysis bounded, inspectable, resumable, and
cost-governed. It does **not** establish that the resulting categories, timelines, or prose are
correct or useful.

The first large private run completed the complete operational pipeline, including embeddings,
representative analysis, taxonomy, synthesis, and book rendering. The result did not meet the
operator's semantic or editorial quality bar. That is an evaluation finding: strong execution
controls did not produce strong analysis.

## Stage design

| Stage | Default provider | Input scope | Output | Principal control |
|---|---|---|---|---|
| Normalize | Local deterministic code | Complete export mapping | Archive IR | ZIP limits, schemas, warnings |
| Represent | Local deterministic code | Declared current paths | Bounded text records | Role/type policy and truncation |
| Embed | Local hashing or OpenAI | Every bounded representation | Numeric vectors | Batch/token limits and cost ledger |
| Baseline profile | Local heuristics | Every representation | Typed conversation profiles | Complete catalog coverage |
| Refine | Structured model, optional | Bounded representatives | Replacement typed profiles | Graph/time-stratified sampling |
| Interpret taxonomy | Structured model, optional | Compressed category dossiers | Category names and hierarchy | Category ceiling and strict schema |
| Synthesize | Structured model, optional | Profiles, references, graph summary | Profiles, timelines, synthesis | Reference reconciliation |
| Render | Local deterministic code | Validated atlas | HTML/PDF draft | No remote subsidiary resources |

The preferred large-archive configuration uses local profiles for every conversation and purchases
structured refinement only for a bounded set of graph-central and time-spanning representatives.
This contains cost, but it also limits semantic coverage. The local baseline synopsis is a shallow
excerpt, refinement is capped, and global requests receive compressed samples rather than every
complete chat.

## Model output is untrusted data

Provider interfaces return strict Pydantic models. Structured Outputs constrain shape; they do not
guarantee factual accuracy, faithful identifiers, good taxonomy, or resistance to instructions
inside archive text.

The pipeline therefore:

- labels archive text as untrusted data in provider instructions;
- disables tools, web search, file search, background mode, and server-side conversations;
- validates every response against a closed schema;
- reconciles final-synthesis category and conversation references against deterministic local IDs;
- rejects or safely reports incomplete, refused, empty, or unparsed responses;
- exposes only stage names and exception classes on private-data failures; and
- creates a review queue for weak or ambiguous assignments.

Identifier reconciliation preserves valid synthesis while discarding unknown references and
supplying conservative local profiles for omitted categories. It is a repair mechanism, not proof
that the retained prose is true.

## Privacy boundary

Core ingestion, compilation, graph construction, local providers, caching, verification, and
rendering make no network calls. OpenAI-enhanced mode is separate and explicitly authorized.

Enhanced mode transmits:

- bounded text from every selected conversation to the embeddings endpoint;
- bounded full representations for the selected refinement set; and
- compressed profile/category dossiers for taxonomy and synthesis.

It uses `store=False`, but that is not equivalent to Zero Data Retention. Source exports, Archive
IR, caches, ledgers, semantic tables, HTML, PDF, and manifests are all sensitive derived data.

## Cost and interruption controls

Before each request, the adapter estimates actual input tokens, reserves the complete configured
output ceiling, and persists the reservation. Successful reported usage settles the reservation;
ambiguous failures retain it. SDK retries are disabled so a hidden retransmission cannot bypass the
ledger.

This is an application-enforced configured-cost ceiling based on a manually recorded price
snapshot. It is not a provider billing guarantee. The ledger has atomic file replacement but no
interprocess lock: use one active process per output directory.

Caches are keyed by source content, provider/model identity, prompt version, and relevant options.
They permit interruption-safe continuation and artifact replay. A fresh hosted-model request remains
nondeterministic.

## Review signals and evidence

“Confidence” values are provider-self-reported or heuristic/composite review signals, not calibrated
probabilities. Conversation keys attached to timeline events provide traceability, not automatic
fact verification. Users must review claims against the source conversations before sharing them.

## What the repository demonstrates

The strongest advanced-AI contribution is the orchestration, not the prose produced by the first
run:

- typed provider protocols and stage-specific routing;
- hybrid local/global analysis with explicit sampling;
- strict structured-output validation;
- model-reference reconciliation;
- privacy and retention gates;
- persistent cost authorization with no hidden retries;
- content-addressed resumable caches;
- safe error surfaces;
- source and artifact provenance;
- deterministic local fallback providers; and
- explicit human-review queues.

See [Evaluation](EVALUATION.md) for the missing quality evidence and
[Reproducibility](REPRODUCIBILITY.md) for replay guarantees.

