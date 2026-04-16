"""Helpers for EOF-submitted multi-line input.

The API is intentionally close to `input()`:

- prompt for one or more lines
- return a single string when the user signals EOF after entering content
- raise `EOFError` when EOF happens at an empty prompt

Policy decisions such as TTY detection, readline setup, retry loops, and non-interactive
stdin handling are left to the caller.
"""

import sys
from typing import Callable, Optional, TextIO


def enable_readline_if_available() -> None:
    try:
        import readline  # noqa: F401
    except ImportError:
        pass


def multiline_input(
    prompt: str = ">>> ",
    continue_prompt: str = "... ",
    input_fn: Callable[[str], str] = input,
    stdout: Optional[TextIO] = None,
) -> str:
    stdout = sys.stdout if stdout is None else stdout

    buffered_lines = []
    current_prompt = prompt
    while True:
        try:
            line = input_fn(current_prompt)
        except EOFError:
            print(file=stdout)
            if not buffered_lines:
                raise
            return "\n".join(buffered_lines) + "\n"

        buffered_lines.append(line)
        current_prompt = continue_prompt
