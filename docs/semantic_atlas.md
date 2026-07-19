# Semantic atlas design

The semantic atlas is the organizational layer between loss-aware Archive IR and the final
thematic book. It treats **subject**, **project**, **conversation purpose**, and **recurring theme**
as different analytical dimensions. A conversation can belong to several categories and projects;
the system never forces every record into one mutually exclusive topic.

## Pipeline

```text
Archive IR
  -> bounded current-path conversation documents
  -> structured conversation profiles
  -> semantic embeddings
  -> nearest-neighbor graph and resolution-controlled communities
  -> named categories, projects, and hierarchy
  -> corpus-level synthesis and review queue
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

Embeddings supply global geometric evidence. A deterministic similarity graph supplies explicit
cross-conversation links. Structured model analysis supplies labels and interpretation. None of
those signals is treated as ground truth in isolation.

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

## Resumability and provenance

Conversation documents have stable content hashes. Expensive results are cached under the selected
run directory's `.semantic_cache` folder by the combination
of document hash, provider, model, prompt version, and relevant options. A restarted run reuses only
compatible complete records and checkpoints one completed provider batch per durable write. Changing
source content or analytical policy invalidates the affected cache rather than silently mixing
generations. Content-free progress callbacks report stage completion and cache hits during long runs.

The final manifest records package and schema versions, non-secret provider configuration, source
fingerprints, artifact hashes, counts, and completion state. It never records credentials or custom
redaction expressions.

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
