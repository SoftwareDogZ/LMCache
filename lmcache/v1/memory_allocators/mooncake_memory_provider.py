# SPDX-License-Identifier: Apache-2.0
"""Mooncake NoF-backed pinned-memory provider."""

# Standard
import ctypes

# First Party
from lmcache.v1.memory_management import PinnedAllocFree
from lmcache.v1.platform import current_device_spec


def create_mooncake_pinned_alloc_free(size: int) -> PinnedAllocFree:
    """Resolve Mooncake NoF alloc/free callbacks for one pinned L1 arena.

    Args:
        size: Number of bytes in the eager L1 arena.

    Returns:
        A resolved allocation pair that allocates with Mooncake and registers
        the resulting host range with the active accelerator runtime.

    Raises:
        RuntimeError: If Mooncake lacks NoF allocator callbacks, allocation
            fails, or the allocated range cannot be pinned.
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

    alloc_callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
    free_callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    raw_alloc = alloc_callback_type(alloc_addr)
    raw_free = free_callback_type(free_addr)

    def alloc() -> int:
        ptr = int(raw_alloc(size) or 0)
        if ptr == 0:
            raise RuntimeError(
                f"Mooncake NoF allocator failed to allocate {size} bytes"
            )
        try:
            pinned = current_device_spec.pin_memory(ptr, size)
        except Exception as exc:
            raw_free(ptr)
            raise RuntimeError(
                "Failed to register Mooncake NoF memory with the accelerator"
            ) from exc
        if not pinned:
            raw_free(ptr)
            raise RuntimeError(
                "Failed to register Mooncake NoF memory with the accelerator"
            )
        return ptr

    def free(ptr: int) -> None:
        unpin_exception: Exception | None = None
        unpinned = False
        try:
            unpinned = current_device_spec.unpin_memory(ptr)
        except Exception as exc:
            unpin_exception = exc
        finally:
            raw_free(ptr)
        if unpin_exception is not None:
            raise RuntimeError(
                "Failed to unregister Mooncake NoF memory from the accelerator"
            ) from unpin_exception
        if not unpinned:
            raise RuntimeError(
                "Failed to unregister Mooncake NoF memory from the accelerator"
            )

    return PinnedAllocFree(
        alloc_fn=alloc,
        alloc_args=(),
        free_fn=free,
        free_args=(),
    )
