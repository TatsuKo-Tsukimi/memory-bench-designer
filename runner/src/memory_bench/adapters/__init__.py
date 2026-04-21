from memory_bench.adapters.base import Adapter, Retrieval
from memory_bench.adapters.recency import RecencyAdapter
from memory_bench.adapters.bm25 import BM25Adapter
from memory_bench.adapters.actr import ACTRAdapter

__all__ = [
    "Adapter",
    "Retrieval",
    "RecencyAdapter",
    "BM25Adapter",
    "ACTRAdapter",
]

# Optional adapters — imported lazily to avoid hard dep on sentence-transformers
try:  # pragma: no cover
    from memory_bench.adapters.embedding import EmbeddingAdapter  # noqa: F401
    from memory_bench.adapters.composite import CompositeAdapter  # noqa: F401

    __all__.extend(["EmbeddingAdapter", "CompositeAdapter"])
except ImportError:
    pass
