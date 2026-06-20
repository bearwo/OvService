from __future__ import annotations

import sys
from typing import Iterator


def print_stream(tokens: Iterator[str], console=None) -> str:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    buf = []
    for token in tokens:
        buf.append(token)
        if console:
            console.print(token, end="", highlight=False)
        else:
            sys.stdout.write(token)
            sys.stdout.flush()
    if not console:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return "".join(buf)
