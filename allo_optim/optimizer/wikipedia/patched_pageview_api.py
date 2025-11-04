"""Patch for pageviewapi to ensure compatibility with Python 3.10+
The pageviewapi library relies on collections attributes that were
moved in Python 3.10+. This patch adds the missing attributes back to
the collections module.

This is a monkey patch and should be used with caution.
Remove as soon as the base library is updated to support Python 3.10+ natively.
"""

import collections
import collections.abc
import sys

if sys.version_info >= (3, 10):
    abcs_to_patch = [
        "Mapping",
        "MutableMapping",
        "Iterable",
        "Callable",
        "Sequence",
        "MutableSequence",
        "Set",
        "MutableSet",
    ]
    for abc_name in abcs_to_patch:
        if not hasattr(collections, abc_name) and hasattr(collections.abc, abc_name):
            setattr(collections, abc_name, getattr(collections.abc, abc_name))

import pageviewapi  # noqa: F401
