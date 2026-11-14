"""Source documentation and static log conventions beyond Ruff's checks."""

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYTHON_ROOTS = ("argus", "api", "scripts", "tests")
LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal", "log"}
CONFIG_FILES = ("Dockerfile.api", "docker-compose.yml", "pyproject.toml", ".env.example", ".gitignore", ".dockerignore")


def _line_violations(text: str, line: int, kind: str, *, content: str | None = None) -> list[str]:
    problems = []
    if len(text) > 120:
        problems.append(f"{line}: {kind} exceeds 120 characters")
    if not (text if content is None else content).isascii():
        problems.append(f"{line}: {kind} contains non-ASCII text")
    return problems


def _python_violations(source: str) -> list[str]:
    """Inspect documentation tokens and log literals without treating application strings as documentation."""
    lines = source.splitlines()
    problems = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            segment = ast.get_source_segment(source, node) or ""
            if node.lineno != node.end_lineno or not (segment.startswith('"""') and segment.endswith('"""')):
                problems.append(f"{node.lineno}: docstring must use one line with triple double quotes")
            problems.extend(_line_violations(lines[node.lineno - 1], node.lineno, "docstring"))
            if not node.value.value.isascii():
                problems.append(f"{node.lineno}: docstring value contains non-ASCII text")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in LOG_METHODS:
            # Inspect static message text, including f-string segments, without restricting interpolated provider data.
            message_index = 1 if node.func.attr == "log" else 0
            messages = list(node.args[message_index : message_index + 1])
            messages.extend(item.value for item in node.keywords if item.arg == "msg")
            for message in messages:
                problems.extend(
                    f"{part.lineno}: static log text contains non-ASCII text"
                    for part in ast.walk(message)
                    if isinstance(part, ast.Constant) and isinstance(part.value, str) and not part.value.isascii()
                )
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            line = token.start[0]
            problems.extend(_line_violations(lines[line - 1], line, "comment", content=token.string))
            if re.search(r"[-\u2500]{5}", token.string):
                problems.append(f"{line}: remove decorative comment separators")
    return problems


def _hash_comment(line: str) -> str | None:
    """Find native hash comments while preserving quoted hashes and escaped literals."""
    quote = None
    escaped = False
    for position, character in enumerate(line):
        if escaped:
            escaped = False
        elif character == "\\" and quote != "'":
            escaped = True
        elif quote:
            if character == quote:
                quote = None
        elif character in {"'", '"'}:
            quote = character
        elif character == "#":
            return line[position:]
    return None


def _config_violations(source: str) -> list[str]:
    problems = []
    for number, line in enumerate(source.splitlines(), 1):
        comment = _hash_comment(line)
        if comment is not None:
            problems.extend(_line_violations(line, number, "configuration comment", content=comment))
            if re.search(r"[-\u2500]{5}", comment):
                problems.append(f"{number}: remove decorative comment separators")
    return problems


def test_python_documentation_and_logs_follow_conventions():
    problems = [
        f"{path.relative_to(ROOT)}:{problem}"
        for package in PYTHON_ROOTS
        for path in sorted((ROOT / package).rglob("*.py"))
        for problem in _python_violations(path.read_text(encoding="utf-8"))
    ]
    assert not problems, "\n".join(problems)


def test_non_python_comments_follow_conventions():
    paths = [ROOT / name for name in CONFIG_FILES] + sorted((ROOT / ".github").rglob("*.yml"))
    problems = [
        f"{path.relative_to(ROOT)}:{problem}"
        for path in paths
        for problem in _config_violations(path.read_text(encoding="utf-8"))
    ]
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "source",
    [
        '"""A summary.\n"""\n',
        '"""' + "x" * 115 + '"""\n',
        "# " + "https://example.com/" + "x" * 120 + "\n",
        "# " + chr(0x2014) + "\n",
        'logger.info("' + chr(0x2192) + '")\n',
        'logger.warning(f"' + chr(0x2192) + ' {value}")\n',
        'logger.log(20, "' + chr(0x2192) + '")\n',
        'logger.error(msg="' + chr(0x2192) + '")\n',
    ],
)
def test_python_checker_rejects_style_violations(source):
    assert _python_violations(source)


def test_python_checker_excludes_assigned_data_and_dynamic_log_values():
    source = (
        '"""A module summary."""\n'
        'PROMPT = """First line.\nSecond line ' + chr(0x2014) + '."""\n'
        'logger.info("Provider response: %s", response)\n'
        'logger.info(f"Provider response: {response}")\n'
        'FIXTURE = "' + chr(0x2192) + '"  # A fixture may retain Unicode data.\n'
    )
    assert _python_violations(source) == []


@pytest.mark.parametrize("source", ["# " + "x" * 119, "# " + chr(0x2014), "# ----- Section -----"])
def test_config_checker_rejects_long_unicode_and_decorative_comments(source):
    assert _config_violations(source)


def test_config_checker_preserves_quoted_hashes_directives_and_environment_examples():
    source = (
        'url = "https://example.com/#fragment"\n# syntax=docker/dockerfile:1\n# ARGUS_API_KEY=\n'
        'value = "' + chr(0x2192) + '" # A configuration value may contain Unicode data.\n'
    )
    assert _config_violations(source) == []
