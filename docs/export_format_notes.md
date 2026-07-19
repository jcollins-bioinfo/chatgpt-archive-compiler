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

Recent multipart exports may provide complete parent pointers while omitting or incompletely
populating redundant child lists. Normalization therefore reconstructs canonical child edges from
parent pointers and retains valid source child declarations as provenance. Differences between the
two representations are reported as one aggregate `source_child_edges_disagree` warning per
conversation; absent child declarations are accepted without warning.

Observed `thoughts` and `reasoning_recap` content is treated as ChatGPT reasoning-interface data,
not as ordinary final-answer text. Both shapes are classified without an unknown-content warning,
kept losslessly in Archive IR, and omitted by the renderer unless the caller deliberately requests
reasoning summaries. The labels do not imply that an export contains raw private chain-of-thought.
