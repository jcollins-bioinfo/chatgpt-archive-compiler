# Roadmap

## v0.1: deterministic ingestion and normalization

Goal: convert a source export ZIP into a normalized Archive IR with warnings and tests.

- [ ] Implement safe ZIP inspection.
- [ ] Build source manifest objects.
- [ ] Locate conversation JSON files.
- [ ] Parse basic conversation records.
- [ ] Reconstruct the primary visible message path where possible.
- [ ] Preserve raw metadata needed for debugging.
- [ ] Add synthetic fixture exports.
- [ ] Add CLI `inspect` command.

## v0.2: local inspection UI

Goal: create a Dash app for upload, validation, and conversation browsing.

- [ ] Add upload/path selection view.
- [ ] Show manifest and warnings.
- [ ] Add searchable conversation table.
- [ ] Add selected-conversation preview.
- [ ] Add include/exclude controls.

## v0.3: redaction and privacy controls

Goal: enable previewable redaction before rendering.

- [ ] Add rule-based detectors for emails, phone numbers, URLs, API-key-like secrets, and custom terms.
- [ ] Add redaction report model.
- [ ] Add manual allow/deny lists.
- [ ] Add before/after preview.
- [ ] Add tests for redaction false-positive and false-negative cases.

## v0.4: rendering MVP

Goal: produce a useful archival PDF and HTML preview.

- [ ] Render HTML preview from Archive IR.
- [ ] Produce cover/title page.
- [ ] Produce table of contents.
- [ ] Render chronological conversations.
- [ ] Render source manifest appendix.
- [ ] Preserve clickable links.
- [ ] Add initial PDF backend.

## v0.5: deterministic analysis

Goal: extract useful corpus-level metadata without external model calls.

- [ ] Conversation counts and date ranges.
- [ ] Message counts by role.
- [ ] Longest conversations.
- [ ] URL/domain inventory.
- [ ] Code block inventory.
- [ ] Timeline views.
- [ ] Approximate token/word counts.

## v1.0: public-quality release candidate

Goal: ready for public GitHub release after security and privacy review.

- [ ] Complete README and screenshots.
- [ ] Complete privacy/security documentation.
- [ ] Add license.
- [ ] Add CI test matrix.
- [ ] Add synthetic demo export.
- [ ] Add build reproducibility manifest.
- [ ] Confirm no real user data appears in repo history.
