import hashlib
from typing import Any, Mapping, Sequence

import torch

try:
    import numpy as np  # optional
except Exception:  # pragma: no cover
    np = None  # type: ignore

# -----------------------------------------------------------------------------
# Device helpers
# -----------------------------------------------------------------------------

def get_device(prefer: str = "cuda") -> torch.device:
    """
    Return a torch.device with a sensible preference order.
    prefer: "cuda" (default) -> "mps" -> "cpu"
    """
    if prefer.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon Metal backend (PyTorch>=1.12)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")  # type: ignore[call-arg]
    return torch.device("cpu")


def sync() -> None:
    """
    Synchronize GPU work, if applicable. No-op on CPU.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):  # type: ignore[attr-defined]
        try:
            torch.mps.synchronize()  # type: ignore[attr-defined]
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Hashing utilities (for caching keys, reproducibility, etc.)
# -----------------------------------------------------------------------------

def _hash_tensor_cpu_fp32(t: torch.Tensor) -> str:
    """
    Backward-compatible tensor hash: fp32 + CPU + shape.
    """
    t32 = t.detach().to(torch.float32).contiguous().cpu()
    h = hashlib.sha1()
    h.update(str(tuple(t32.shape)).encode("utf-8"))
    h.update(t32.numpy().tobytes())
    return h.hexdigest()


def hash_tensor(t: torch.Tensor, *, dtype: torch.dtype = torch.float32, with_shape: bool = True) -> str:
    """
    Deterministic hash of a tensor. Moves to CPU, casts to `dtype`, and hashes bytes.
    """
    t_ = t.detach().to(dtype).contiguous().cpu()
    h = hashlib.sha1()
    if with_shape:
        h.update(str(tuple(t_.shape)).encode("utf-8"))
    h.update(t_.numpy().tobytes())
    return h.hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def hash_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _to_bytes(obj: Any) -> bytes:
    """
    Convert common Python/NumPy/Torch objects to deterministic bytes for hashing.
    """
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, str):
        return obj.encode("utf-8")
    if isinstance(obj, torch.Tensor):
        t = obj.detach().to(torch.float32).contiguous().cpu()
        return (str(tuple(t.shape)).encode("utf-8") + b"|" + t.numpy().tobytes())
    if np is not None and isinstance(obj, np.ndarray):  # type: ignore[misc]
        arr = np.asarray(obj, dtype=np.float32)
        return (str(tuple(arr.shape)).encode("utf-8") + b"|" + arr.tobytes(order="C"))
    if isinstance(obj, Mapping):
        chunks = []
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            chunks.append(_to_bytes(k))
            chunks.append(b":")
            chunks.append(_to_bytes(obj[k]))
            chunks.append(b";")
        return b"{" + b"".join(chunks) + b"}"
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return b"[" + b",".join(_to_bytes(x) for x in obj) + b"]"
    return str(obj).encode("utf-8")


def hash_obj(obj: Any) -> str:
    """
    Deterministic SHA1 of an arbitrary (supported) object.
    """
    return hashlib.sha1(_to_bytes(obj)).hexdigest()

# -----------------------------------------------------------------------------
# Lightweight in-memory cache (module-level)
# -----------------------------------------------------------------------------

_GENERIC_CACHE: dict[str, Any] = {}

# Backward-compatibility alias (some code may still import this name)
_PADDING_RESULT_CACHE = _GENERIC_CACHE


def cache_get(key: str, default: Any = None) -> Any:
    return _GENERIC_CACHE.get(key, default)


def cache_set(key: str, value: Any) -> None:
    _GENERIC_CACHE[key] = value


def cache_contains(key: str) -> bool:
    return key in _GENERIC_CACHE


def cache_clear() -> None:
    _GENERIC_CACHE.clear()