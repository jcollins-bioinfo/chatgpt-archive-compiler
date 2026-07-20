# Semantic atlas design

The semantic atlas is the organizational layer between loss-aware Archive IR and the final
thematic book. It treats **subject**, **project**, **conversation purpose**, and **recurring theme**
as different analytical dimensions. A conversation can belong to several categories and projects;
the system never forces every record into one mutually exclusive topic.

## Pipeline

```text
Archive IR
  -> bounded current-path conversation documents
  -> semantic embeddings
  -> transparent local profiles for every conversation
  -> nearest-neighbor graph and resolution-controlled communities
  -> bounded central/earliest/latest representative refinement
  -> named categories, projects, and hierarchy
  -> compressed corpus-level synthesis and review queue
  -> semantic atlas artifacts and thematic book
```

Only user and assistant messages on the declared current path are represented by default. Alternate
branches remain preserved in Archive IR but do not silently affect categorization. Thinking traces,
reasoning summaries, system messages, tool messages, referenced files, and remote assets are also
excluded unless a caller deliberately changes the policy.

## Analytical dimensions

Each enhanced conversation profile can describe:

- a concise synopsis;
- primary and secondary subjects;
- continuing projects;
- conversation purposes such as research, implementation, troubleshooting, planning, reflection,
  creative work, or practical decision-making;
- named entities and important artifacts;
- goals, decisions, actions, and unresolved questions;
- temporal function within a thread, such as initiation, continuation, refinement, reversal, or
  resolution; and
- assignment confidence and ambiguity notes.

Confidence values are provider-self-reported or heuristic/composite review signals, not calibrated
probabilities. Conversation references make claims traceable; they do not establish factual support
without human comparison to the source.

Embeddings supply global geometric evidence. A deterministic similarity graph supplies explicit
cross-conversation links. Structured model analysis supplies labels and interpretation. None of
those signals is treated as ground truth in isolation.

For large archives, the preferred pipeline does not purchase a generative profile for every
conversation. Deterministic local profiles establish complete catalog coverage before graph
discovery. Each graph category then contributes a small, fair set of central and time-spanning
representatives for model refinement. Taxonomy and final synthesis requests receive compact
profiles and category dossiers, so structured-model work scales with the bounded category count
rather than total conversation count.

The default synthesis ceiling is 64 leaf categories. When graph discovery produces more, community
centroids are consolidated deterministically before model interpretation so the structured global
passes remain bounded. Every consolidation is disclosed as `consolidated_category` in the review
queue; changing `max_leaf_categories` changes this explicit quality/scale tradeoff.

## Privacy boundary

Deterministic document construction, cache validation, graph construction, artifact writing, and
book rendering are local operations. Enhanced analysis is an explicit opt-in network operation: the
bounded conversation representations supplied to the configured provider leave the Colab runtime.
The real-data notebook must show the selected fields and estimated token volume before asking for a
separate acknowledgment. API credentials are read from Colab secrets and are never written to Drive,
logs, manifests, or Git.

Private products are written only to the requested output directory. Logs contain counts, stable
stage names, model identifiers, hashes, and paths; they do not contain titles, message text, entity
names, prompts, response bodies, source identifiers, or source-derived exception messages.

The OpenAI adapter uses Responses structured outputs with `store=False` and does not enable tools,
web search, files, background mode, or server-side conversations. According to OpenAI's current
[data-control documentation](https://developers.openai.com/api/docs/guides/your-data), API data is
not used to train OpenAI models unless the account explicitly opts in, but default abuse-monitoring
logs may retain prompts and responses for up to 30 days. `store=False` prevents the separate
Responses application-state retention; it does not by itself confer Zero Data Retention. The Colab
acknowledgment states this distinction directly.

## API cost boundary

The budgeted notebook uses a shared persistent `ApiBudget` for embeddings, representative
profiling, taxonomy interpretation, and final synthesis. Before each network request, the provider
adapter tokenizes the actual payload and structured-output schema, adds a conservative input
allowance, reserves the request's complete configured `max_output_tokens` cost, and atomically
writes that reservation to `api_budget_ledger.json`. A request is not sent if its reservation would
exceed the remaining configured ceiling.

Reported provider usage replaces the conservative reservation after a successful response. Failed,
interrupted, or usage-unobservable requests retain their reservation, preventing a restarted Colab
session from silently resetting authorization. The ledger contains only stages, models, token
counts, prices, and costs. Pricing remains explicit configuration because an application-side
ledger cannot detect a future provider price change; the notebook records the price snapshot date
and requires the complete scheduled plan to fit the ceiling before retrieving an API key.

The ledger uses atomic replacement but no interprocess lock. Exactly one process may write a given
ledger/output directory at a time. Its ceiling uses configured price assumptions and is not a
provider billing guarantee.

Budgeted providers also disable SDK-level automatic retries. An ambiguous failure therefore retains
one reservation; any later transmission must be initiated deliberately and pass the persistent
ledger again. A resumed ledger rejects ceiling changes by default. The notebook may explicitly
increase—but never decrease—the ceiling only after the user enters the new exact dollar
authorization; the existing charged history is preserved.

Structured Outputs constrain response shape, but generated category and conversation identifiers can
still drift from the supplied semantic values. Final synthesis therefore reconciles identifiers
against the deterministic local taxonomy: unknown references are removed, duplicate profiles are
collapsed, and an omitted category receives a conservative profile from cached analyses. The global
model synthesis and every valid model-authored profile remain intact. Responses that are incomplete,
content-filtered, refused, or unparsed produce stage-specific content-free diagnostics following
OpenAI's documented response states.

The [embeddings endpoint](https://developers.openai.com/api/reference/resources/embeddings/methods/create)
limits each input to 8,192 tokens and each complete request to 300,000 tokens across all inputs. The
OpenAI adapter therefore partitions item-count batches again at a 280,000-token internal ceiling.
The budgeted notebook also applies conservative local tokens-per-minute pacing, avoiding predictable
rate-limit failures before transmission.

## Resumability and provenance

Conversation documents have stable content hashes. Expensive results are cached under the selected
run directory's `.semantic_cache` folder by the combination
of document hash, provider, model, prompt version, and relevant options. A restarted run reuses only
compatible complete records and checkpoints one completed provider batch per durable write. Changing
source content or analytical policy invalidates the affected cache rather than silently mixing
generations. Baseline and representative profiles use separate cache files so one cannot invalidate
the other in a resume loop. Content-free progress callbacks report stage completion and cache hits
during long runs.

The final manifest records package and schema versions, non-secret provider configuration, source
fingerprints, artifact hashes, counts, and completion state. It never records credentials or custom
redaction expressions.

## Observed quality boundary

The first large private run completed operationally but did not meet the operator's semantic or
editorial quality bar. The preferred cost-controlled design profiles most chats with a shallow local
synopsis, caps external refinement at 144 conversations by default, samples category dossiers for
global passes, and truncates bounded representations. These controls contain cost but can miss
long-range context, subtle project continuity, and important details outside excerpts.

Engineering validation therefore must not be presented as semantic validation. See
[EVALUATION.md](EVALUATION.md) for the required benchmark and promotion gate.

## Human review

The atlas is a proposed organization, not an irreversible rewrite. Low-confidence profiles and
assignments, isolated conversations, secondary-category ambiguity, singleton categories, and any
scale-driven community consolidation enter a review queue. These signals identify where human
judgment is most valuable without presenting the taxonomy as ground truth. A durable
rename/merge/split override format is a planned extension; it is not silently simulated in this
first atlas draft.

## Book design

The thematic book is organized by category chapters, with chronology preserved inside each chapter.
It uses self-contained print CSS, local system fonts, running headers, page numbers, chapter-opening
pages, a hierarchical contents section, conversation metadata, restrained color, and explicit
cross-references. No web fonts or remote images are fetched. HTML remains the inspectable canonical
rendering; PDF is a derived artifact produced locally with WeasyPrint. Every conversation remains in
its category directory, while global and per-category profile budgets plus project/event limits keep
the default PDF selective and bounded. Any omitted project material remains available in the
structured atlas artifacts and is disclosed in the book.

