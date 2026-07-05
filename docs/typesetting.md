# Typesetting Notes

The final PDF should be treated as a book or technical monograph, not a generic report.

## Required capabilities

- Cover/title page.
- Copyright or publication-status page.
- Table of contents.
- Internal links.
- Conversation anchors.
- Role-aware transcript rendering.
- Code block rendering.
- URL index.
- Conversation index.
- Source manifest appendix.
- Redaction report appendix.

## Candidate backends

### HTML preview

Useful for fast iteration and Dash integration. The HTML renderer should be the first renderer implemented.

### LaTeX or Typst final renderer

The final book-quality PDF renderer should be separated from the HTML preview. LaTeX or Typst should receive a structured document model rather than raw conversation JSON.

## Open questions

- Should the default archive be chronological, thematic, or both?
- Should long conversations be split into chapters or sections?
- Should tool outputs be hidden, summarized, or rendered in full?
- Should code blocks be indexed by language?
