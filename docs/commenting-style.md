# Comments and docstrings

Write for someone with decent Python knowledge and moderate familiarity with the domain.
Let names, types, and straightforward code explain the routine parts. Add documentation when
it supplies information the reader would otherwise have to infer or rediscover.

## Source documentation

Code lines have a 120-character limit. Every retained docstring occupies one physical line,
including the opening and closing triple
double quotes. Keep the entire line within 120 characters, including indentation. Shorten the
wording to fit. Use a brief summary without sections, examples, or parameter lists.

Function summaries use imperative wording and end with a period:

```python
def rrf_merge(result_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Fuse ranked lists with RRF, merging identical text and retaining the first metadata."""
```

Module and class summaries describe their purpose:

```python
"""Qdrant collection setup and dense retrieval."""
```

Keep short descriptions for substantive modules. Functions, methods, classes, constructors,
and test helpers may omit docstrings when their names and code are sufficient. Avoid adding
a summary merely to repeat a name or type annotation. Public visibility alone does not require
a docstring.

Comments also occupy a single physical line, within 120 characters. Use them for non-obvious
reasoning, constraints, or workarounds. Put them immediately above the relevant code; trailing
comments are suitable for brief constant or configuration annotations.

```python
# Remove hidden ix:header metadata while preserving visible figures in other ix:* tags.
html_cleaned = re.sub(r"<ix:header\b[^>]*>.*?</ix:header>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
```

Use plain English and ASCII in source comments, docstrings, and static log text. Remove narration,
decorative separators, commented-out code, and redundant field descriptions. A TODO or FIXME
must identify a specific outstanding issue. Keep required tool directives such as `noqa`
where they apply.

These conventions apply throughout `argus/`, `api/`, `scripts/`, and `tests/`, and to all project
languages and configuration, including Dockerfiles, YAML, TOML, environment examples, and ignore
files. Use native comment syntax, keeping each explanation on one physical line within 120
characters. Preserve required syntax directives. Commented environment assignments are
copyable examples rather than dead code. Multiline prompts, HTML, CSS, CLI descriptions, and
test fixtures are application data rather than docstrings. Markdown prose may use longer
explanations and Unicode. Dynamic provider data interpolated into logs retains its original text.

## Longer explanations

Put detailed behavior and design decisions in a focused Markdown guide under `docs/`.
Use descriptive headings, short paragraphs, and concrete examples when they help. Tables
work well for comparisons and mappings. Explain what the code does and why a decision matters;
avoid promotional claims, repeated summaries, and a page for every function.

Do not turn a paragraph into a stack of single-line comments to fit the source style.
A short comment may link to a guide when the connection would otherwise be difficult to find.
Keep the [documentation index](README.md) and the README link useful as the collection grows.

## Checks

Run the same lint and formatting checks as CI:

```bash
ruff check .
ruff format --check .
pytest tests/test_source_conventions.py
```

The Ruff configuration disables missing-docstring rules (`D1`). Retained docstrings still
receive checks for imperative function summaries, punctuation, quoting, and spacing. `W505`
sets the documentation line limit to 120 characters; `ERA001` catches commented-out code.
`RUF002` and `RUF003` detect ambiguous Unicode characters in docstrings and comments.

Ruff's `D200` catches a short summary spread unnecessarily across lines. Ruff does not prohibit
all multiline docstrings or wrapped comment paragraphs, and `W505` has exceptions for such
things as standalone URLs. Check relevance and wrapped paragraphs during review. The convention
tests in `tests/test_source_conventions.py` enforce single-line ASCII comments and docstrings,
the strict documentation line limit, ASCII static log text, and non-Python configuration
comments. They cover all Python packages and tests while excluding application strings.
