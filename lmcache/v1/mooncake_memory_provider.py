# SPDX-License-Identifier: Apache-2.0
"""Mooncake NoF-backed pinned-memory provider for in-process mode."""

# Standard
import ctypes

# First Party
from lmcache import torch_dev
from lmcache.v1.memory_management import PinnedAllocFree


def create_mooncake_pinned_alloc_free(size: int) -> PinnedAllocFree:
    """Create callbacks for one Mooncake NoF-backed LocalCPUBackend arena.

    Args:
        size: Number of bytes in the eager LocalCPUBackend arena.

    Returns:
        Allocation callbacks that allocate through Mooncake and register the
        resulting host range with CUDA.

    Raises:
        RuntimeError: If the Mooncake callbacks or CUDA host registration are
            unavailable, or if allocation or registration fails.
    """
    try:
        # Third Party
        from mooncake.store import get_alloc_func_addr, get_free_func_addr
    except ImportError as exc:
        raise RuntimeError(
            "Mooncake NoF L1 allocation requires mooncake.store with "
            "get_alloc_func_addr/get_free_func_addr"
        ) from exc

    alloc_addr = int(get_alloc_func_addr())
    free_addr = int(get_free_func_addr())
    if alloc_addr == 0 or free_addr == 0:
        raise RuntimeError(
            "Mooncake NoF allocator callbacks are unavailable; build Mooncake "
            "with USE_NOF enabled"
        )

    if (
        torch_dev is None
        or not torch_dev.is_available()
        or not hasattr(torch_dev, "cudart")
    ):
        raise RuntimeError(
            "Mooncake NoF L1 allocation requires CUDA host registration support"
        )

    alloc_callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
    free_callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    raw_alloc = alloc_callback_type(alloc_addr)
    raw_free = free_callback_type(free_addr)
    try:
        cudart = torch_dev.cudart()
    except Exception as exc:
        raise RuntimeError(
            "Mooncake NoF L1 allocation cannot initialize the CUDA runtime"
        ) from exc

    def alloc(requested_size: int) -> int:
        if requested_size != size:
            raise RuntimeError(
                "Mooncake NoF allocator size changed after provider creation: "
                f"expected {size}, got {requested_size}"
            )
        ptr = int(raw_alloc(requested_size) or 0)
        if ptr == 0:
            raise RuntimeError(
                f"Mooncake NoF allocator failed to allocate {requested_size} bytes"
            )
        try:
            result = cudart.cudaHostRegister(ptr, requested_size, 0)
            if int(result) != 0:
                raise RuntimeError(f"cudaHostRegister returned error {int(result)}")
        except Exception as exc:
            raw_free(ptr)
            raise RuntimeError(
                "Failed to register Mooncake NoF memory with CUDA"
            ) from exc
        return ptr

    def free(ptr: int) -> None:
        unregister_exception: Exception | None = None
        try:
            result = cudart.cudaHostUnregister(ptr)
            if int(result) != 0:
                unregister_exception = RuntimeError(
                    f"cudaHostUnregister returned error {int(result)}"
                )
        except Exception as exc:
            unregister_exception = exc
        finally:
            raw_free(ptr)
        if unregister_exception is not None:
            raise RuntimeError(
                "Failed to unregister Mooncake NoF memory from CUDA"
            ) from unregister_exception

    return PinnedAllocFree(
        alloc_fn=alloc,
        alloc_args=(),
        free_fn=free,
        free_args=(),
    )
