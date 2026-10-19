"""Compatibility shims for running current langmem on Python 3.10.

Re-exports from E1 compat.
"""

import typing

import typing_extensions

if not hasattr(typing, "NotRequired"):  # pragma: no cover - py3.10 only
    typing.NotRequired = typing_extensions.NotRequired
    typing.NotRequired.__module__ = "typing"
