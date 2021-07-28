# pylint: disable=unused-import
# flake8: noqa: F401

try:
    # Python 3.8+
    from typing import Protocol
except ImportError:
    Protocol = object  # type: ignore
