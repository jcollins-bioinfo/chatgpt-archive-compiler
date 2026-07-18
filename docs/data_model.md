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

`Conversation.current_path_node_ids` is derived only by following parent pointers from the declared
`current_node_id`. Missing or invalid current-node provenance produces an empty visible path and a
warning; the normalizer never concatenates sibling branches or guesses a transcript.

## Schema drift

Recognized fields receive typed representations. Unknown valid JSON fields are retained in
`source_extras`; unsupported content becomes an `UNKNOWN` block with its JSON structure intact.
Recoverable anomalies use stable warning codes and JSON Pointer locations. Strict mode rejects any
such warning, while tolerant mode preserves interpretable data.

All IR models forbid undeclared fields and suppress input values in validation error text.
