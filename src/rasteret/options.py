# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Global runtime options for Rasteret.

Rasteret's public read APIs accept per-call overrides (e.g. ``progress=``),
but some behaviors are easier to control globally in notebooks and scripts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Iterator


@dataclass(frozen=True)
class RasteretOptions:
    """Runtime options that control default behavior."""

    progress: bool = False


_OPTIONS = RasteretOptions()


def get_options() -> RasteretOptions:
    """Return the current global options."""
    return _OPTIONS


def set_options(*, progress: bool | None = None) -> None:
    """Update global options.

    Parameters
    ----------
    progress:
        Default for progress bars in read APIs. When ``None``, keep the
        current value.
    """
    global _OPTIONS
    next_options = _OPTIONS
    if progress is not None:
        next_options = replace(next_options, progress=bool(progress))
    _OPTIONS = next_options


@contextmanager
def options(*, progress: bool | None = None) -> Iterator[None]:
    """Temporarily override global options.

    Examples
    --------
    >>> import rasteret
    >>> with rasteret.options(progress=True):
    ...     arr = col.get_numpy(..., bands=["B02"])
    """
    global _OPTIONS
    old = _OPTIONS
    try:
        set_options(progress=progress)
        yield
    finally:
        _OPTIONS = old
