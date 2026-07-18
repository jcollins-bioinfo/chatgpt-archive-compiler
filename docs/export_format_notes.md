# Export format notes

The source format is treated as semi-stable rather than permanent.

Payload selection uses deterministic precedence: root `conversations.json`, then exactly one
case-insensitive basename match, then conversation-like JSON filenames. Multiple root-level files
matching `conversations-<number>.json` are recognized as one multipart payload only when their
numeric indices are unique, contiguous, and start at zero. Parts are consumed incrementally in
numeric order. Other equally ranked matches are fatal rather than guessed.

Canonical payload roots are conversation lists. A `{ "conversations": [...] }` or
`{ "items": [...] }` wrapper and a single mapping-bearing conversation object are accepted with a
schema-drift warning. JSON must be strict UTF-8; a UTF-8 BOM is accepted and recorded. Non-finite
JSON constants are rejected.

ZIP input is never extracted. Metadata preflight rejects unsafe paths, duplicate/case-colliding
paths, symlinks, encryption, excessive archive/member sizes, and suspicious compression ratios
before any member is decompressed. Per-member and combined selected-JSON sizes are enforced again
while streaming. Conversation, graph-node, duplicate-ID, and warning limits are shared across all
multipart members.
