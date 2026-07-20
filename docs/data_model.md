# Archive IR v1

Archive IR v1 is the canonical, renderer-independent representation produced by
`ingest_export_zip`.

## Provenance

`SourceManifest` records the source filename, compressed size, optional source-ZIP digest, member
metadata, aggregate declared sizes, and the ordered selected conversation payload paths. A
single-part export also populates the backward-compatible singular path and digest fields; every
selected member carries its own SHA-256 digest in the file manifest. Absolute host paths and
wall-clock ingestion times are intentionally excluded so identical inputs and configuration produce
identical IR bytes.

## Conversation graph

Every mapping entry becomes a `MessageNode`, including message-less structural roots and nodes on
alternate or regenerated branches. The mapping key is the authoritative `node_id`. A semantic
`Message.message_id` remains a separate identifier.

Parent pointers are authoritative for graph topology. `MessageNode.children_node_ids` is rebuilt
deterministically from those pointers, which supports exports that omit or incompletely populate
redundant `children` arrays. A valid source declaration is preserved separately in
`source_children_node_ids`; `None` distinguishes a missing or malformed declaration from an
explicitly empty list. Disagreement is summarized once per conversation rather than once per edge.

`Conversation.current_path_node_ids` is derived only by following parent pointers from the declared
`current_node_id`. Missing or invalid current-node provenance produces an empty visible path and a
warning; the normalizer never concatenates sibling branches or guesses a transcript.

## Schema drift

Recognized fields receive typed representations. Unknown valid JSON fields are retained in
`source_extras`; unsupported content becomes an `UNKNOWN` block with its JSON structure intact.
Recoverable anomalies use stable warning codes and JSON Pointer locations. Strict mode rejects any
such warning, while tolerant mode preserves interpretable data.

The observed source content types `thoughts` and `reasoning_recap` are intentionally represented as
`THINKING_TRACE` and `REASONING_SUMMARY`. Their complete source object remains in block metadata
because OpenAI does not publish a field-level export schema for these records. They are distinct
from ordinary assistant answers and are excluded from rendered documents unless explicitly enabled.

All IR models forbid undeclared fields and suppress input values in validation error text.
