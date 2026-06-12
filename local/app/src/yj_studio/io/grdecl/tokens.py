"""Low-level GRDECL tokeniser.

GRDECL is whitespace-delimited ASCII with these quirks worth knowing
before you read the code:

- Line comments start with ``--`` and run to end-of-line.
- Numeric data uses ECLIPSE RLE: ``56064*0`` means "56064 zeros". The
  multiplier and value are both integers (or one float). ``*`` never
  appears as a stand-alone token.
- Records end with a single ``/`` on its own — i.e. each keyword's
  payload terminates with a slash.
- File headers are written in the OS code page (GBK on Chinese Windows)
  but the data payload is pure ASCII numbers, so we open with
  ``encoding='gbk', errors='replace'`` and the lossy header decode
  doesn't matter.
- File sizes routinely exceed several GB. We must iterate line-by-line
  and never materialise the whole token stream.

Public functions in this module yield raw tokens or expanded
(value, count) pairs — higher-level keyword parsing lives in
``parser.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Iterator


def open_text(path: Path) -> IO[str]:
    # GBK first (matches Petrel on zh-CN Windows); fall back to latin-1
    # which never raises since both are single-byte-clean on data lines.
    try:
        return open(path, "r", encoding="gbk", errors="replace")
    except Exception:
        return open(path, "r", encoding="latin-1", errors="replace")


def strip_comment(line: str) -> str:
    idx = line.find("--")
    return line if idx < 0 else line[:idx]


def iter_tokens(path: Path) -> Iterator[str]:
    """Yield whitespace-separated tokens, skipping comments and blanks.

    The returned tokens are raw strings — RLE pairs like ``56064*0``
    are yielded as a single token. Use :func:`iter_expanded` if you
    want the multiplier already applied.
    """

    with open_text(path) as f:
        for raw in f:
            line = strip_comment(raw).strip()
            if not line:
                continue
            for tok in line.split():
                yield tok


def parse_rle(token: str) -> tuple[int, str]:
    """Return ``(count, value_token)`` for an RLE pair, or ``(1, token)``.

    Does not parse the value — callers know whether it's int or float.
    """

    star = token.find("*")
    if star < 0:
        return 1, token
    return int(token[:star]), token[star + 1 :]


def iter_keyword_floats(path: Path, keyword: str) -> Iterator[float]:
    """Stream every float value of a single GRDECL keyword.

    The iterator starts after we see ``keyword`` (case-insensitive,
    as its own token) and stops when we see a stand-alone ``/`` or
    another keyword token.

    A *keyword token* is any alphabetic-leading uppercase identifier
    such as ``ZCORN``, ``COORD``, ``ACTNUM``, ``MAPAXES``. We detect
    them with a coarse rule: token starts with a letter. ECLIPSE
    keywords are always uppercase ASCII; numeric tokens always start
    with a digit, ``+``, ``-`` or ``.``.

    RLE pairs are expanded — yields ``count`` copies of the value.
    """

    target = keyword.upper()
    it = iter_tokens(path)
    # Skip to keyword
    for tok in it:
        if tok.upper() == target:
            break
    else:
        raise KeyError(f"keyword {keyword!r} not found in {path}")
    # Yield payload
    for tok in it:
        if tok == "/":
            return
        # Defensive: if we hit another keyword without seeing '/',
        # the file is malformed but we stop cleanly rather than yield
        # garbage from an unrelated section.
        if tok[:1].isalpha():
            return
        count, value_tok = parse_rle(tok)
        value = float(value_tok)
        for _ in range(count):
            yield value


def iter_keyword_ints(path: Path, keyword: str) -> Iterator[int]:
    """Same as :func:`iter_keyword_floats` but yields ints.

    Used for ACTNUM and integer property keywords like LITHOLOGIES.
    """

    target = keyword.upper()
    it = iter_tokens(path)
    for tok in it:
        if tok.upper() == target:
            break
    else:
        raise KeyError(f"keyword {keyword!r} not found in {path}")
    for tok in it:
        if tok == "/":
            return
        if tok[:1].isalpha():
            return
        count, value_tok = parse_rle(tok)
        value = int(value_tok)
        for _ in range(count):
            yield value
