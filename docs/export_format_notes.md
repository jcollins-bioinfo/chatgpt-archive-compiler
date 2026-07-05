# Export Format Notes

The parser must treat the source export format as semi-stable rather than fixed. The first implementation should support common conversation JSON layouts while preserving unknown fields and emitting warnings.

## Expected file candidates

- `conversations.json`
- numbered or split conversation JSON files
- shared conversation metadata JSON files
- auxiliary metadata files

## Parser requirements

- Locate likely conversation files from the ZIP manifest.
- Avoid blind extraction.
- Decode JSON with clear error reporting.
- Preserve raw records in metadata during early development.
- Reconstruct the primary visible path when conversations are represented as a message graph.
- Record unresolved branches or orphan nodes as warnings or future sidecars.

## Synthetic fixtures

Tests must use synthetic exports only. Fixtures should cover:

- minimal valid export;
- missing title;
- empty conversation;
- branching/regenerated messages;
- malformed message content;
- multiple source files;
- unsafe ZIP path entries.
