# Comment conventions

## Module docstrings

Every `.py` file opens with a docstring: path line, one-line purpose, then
`Responsibilities:` and `Not responsible for:` bullet lists. See
`argus/orchestration/governor.py` for the canonical shape.

## Function & class docstrings

Google style. One-line summary, then `Args:` / `Returns:` / `Raises:` where
applicable. Required on every public (non-underscore) `def` and `class`.
Private `_helpers` get one only when the name and signature don't already
make the behavior obvious.

Test functions follow the same rule: a one-line docstring stating the
property under test, not a restatement of the test's name.

## Inline `#` comments

Why-only, and sparse. A comment earns its place only where the code would
otherwise read as arbitrary or wrong — a non-obvious constraint, a unit
conversion, an upstream quirk being worked around. Don't comment what the
code already says.

Sentence case, no trailing period, wrapped at the ruff `line-length = 100`.
Condense multi-line blocks to the load-bearing sentence:

```python
# Daily bars would mismatch the resolution the technical agent expects
```

rather than restating the surrounding logic across several lines.

## References

Code-local only. Pointing at a sibling module is fine (`see
argus/data/fetchers.py`). Don't cite ADRs or narrate incident history from
source comments — that belongs in `docs/adr/` and PR descriptions, which can
carry it without going stale independently of the code around them.

## Commit messages

No AI references — no "Co-Authored-By: Claude", no mention of Claude,
Anthropic, or AI assistance anywhere in the commit message.
