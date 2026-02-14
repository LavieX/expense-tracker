"""Parser registry for bank CSV parsers.

Each parser is a module exposing a ``parse(file_path, institution, account)``
function that returns a :class:`~expense_tracker.models.StageResult`.  The
``PARSERS`` dict maps parser names (used in config) to parse functions, and
``get_parser()`` provides a convenient lookup with a clear error on unknown
names.
"""

from __future__ import annotations

from collections.abc import Callable

from expense_tracker.parsers import capital_one, chase, elevations

PARSERS: dict[str, Callable] = {
    "chase": chase.parse,
    "capital_one": capital_one.parse,
    "elevations": elevations.parse,
}


def get_parser(name: str) -> Callable:
    """Look up a parser by name.

    Args:
        name: Parser name as specified in account config, e.g. "chase".

    Returns:
        The parse function for the named parser.

    Raises:
        KeyError: If no parser is registered under the given name.
    """
    return PARSERS[name]
