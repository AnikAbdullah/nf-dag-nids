from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from hashlib import blake2b

import numpy as np

_MAX_CACHE_SIZE = 32768
_SUBSET_CACHE: OrderedDict[tuple[tuple[int, ...], tuple[float, ...], str], float] = OrderedDict()
_BACKGROUND_DIGEST_CACHE: OrderedDict[tuple[int, tuple[int, ...], str], str] = OrderedDict()


def background_digest(background: np.ndarray) -> str:
    """Return a stable digest for a background matrix.

    The digest is cached by object id because Module 5a repeatedly evaluates
    many coalitions against the same benign reference matrix.
    """
    arr = np.ascontiguousarray(np.asarray(background, dtype=np.float64))
    key = (id(background), arr.shape, arr.dtype.str)
    cached = _BACKGROUND_DIGEST_CACHE.get(key)
    if cached is not None:
        _BACKGROUND_DIGEST_CACHE.move_to_end(key)
        return cached

    h = blake2b(digest_size=16)
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.view(np.uint8))
    digest = h.hexdigest()

    _BACKGROUND_DIGEST_CACHE[key] = digest
    if len(_BACKGROUND_DIGEST_CACHE) > 64:
        _BACKGROUND_DIGEST_CACHE.popitem(last=False)
    return digest


def cached_subset_eval(
    subset: frozenset[int],
    x_tuple: tuple[float, ...],
    background_hash: str,
    eval_fn: Callable[[frozenset[int]], float],
) -> float:
    """LRU-cached wrapper around coalition value evaluations.

    Cache key: (subset, x_tuple, background_hash). The caller should include
    DAG/method/sample-size fingerprints in background_hash when those change.
    """
    key = (tuple(sorted(subset)), x_tuple, background_hash)
    cached = _SUBSET_CACHE.get(key)
    if cached is not None:
        _SUBSET_CACHE.move_to_end(key)
        return cached

    value = float(eval_fn(subset))
    _SUBSET_CACHE[key] = value
    if len(_SUBSET_CACHE) > _MAX_CACHE_SIZE:
        _SUBSET_CACHE.popitem(last=False)
    return value


def clear_subset_cache() -> None:
    """Clear Module 5a subset and background digest caches."""
    _SUBSET_CACHE.clear()
    _BACKGROUND_DIGEST_CACHE.clear()


def subset_cache_info() -> dict[str, int]:
    """Small introspection helper for tests and notebooks."""
    return {
        "subset_entries": len(_SUBSET_CACHE),
        "background_entries": len(_BACKGROUND_DIGEST_CACHE),
    }
