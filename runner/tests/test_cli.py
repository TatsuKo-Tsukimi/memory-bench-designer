from __future__ import annotations

import pytest

import click

from memory_bench.cli import _parse_seeds


def test_parse_seeds_rejects_invalid_seed():
    with pytest.raises(click.UsageError, match="invalid seed"):
        _parse_seeds("abc", 42)


def test_parse_seeds_uses_default_when_missing():
    assert _parse_seeds(None, 42) == [42]
    assert _parse_seeds("", 42) == [42]
    assert _parse_seeds("1,2,3", 42) == [1, 2, 3]
