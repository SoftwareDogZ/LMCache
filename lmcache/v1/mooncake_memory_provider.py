# SPDX-License-Identifier: Apache-2.0
"""Mooncake-backed pinned-memory provider for in-process mode."""

# Standard
from typing import Protocol
import ctypes
import os

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.v1.memory_management import PinnedAllocFree

_ACL_SUCCESS = 0
_ACL_HOST_REGISTER_MAP = 0x2
_ACL_HOST_REGISTER_TYPE = 0x10000000


class _HostMemoryRegistrar(Protocol):
    def register(self, ptr: int, size: int) -> None: ...

    def unregister(self, ptr: int) -> None: ...


class _CudaHostMemoryRegistrar:
    def __init__(self) -> None:
        if (
            torch_dev is None
            or not torch_dev.is_available()
            or not hasattr(torch_dev, "cudart")
        ):
            raise RuntimeError("CUDA host registration support is unavailable")
        try:
            self.cudart = torch_dev.cudart()
        except Exception as exc:
            raise RuntimeError("Cannot initialize the CUDA runtime") from exc

    def register(self, ptr: int, size: int) -> None:
        result = self.cudart.cudaHostRegister(ptr, size, 0)
        if int(result) != 0:
            raise RuntimeError(f"cudaHostRegister returned error {int(result)}")

    def unregister(self, ptr: int) -> None:
        result = self.cudart.cudaHostUnregister(ptr)
        if int(result) != 0:
            raise RuntimeError(f"cudaHostUnregister returned error {int(result)}")


def _load_ascendcl() -> ctypes.CDLL:
    candidates = ["libascendcl.so"]
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        candidates.extend(
            [
                os.path.join(ascend_home, "lib64", "libascendcl.so"),
                os.path.join(ascend_home, "runtime", "lib64", "libascendcl.so"),
            ]
        )
    candidates.append("/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so")

    errors: list[str] = []
    for candidate in dict.fromkeys(candidates):
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(
        "Cannot load libascendcl.so for ACL host registration: " + "; ".join(errors)
    )


class _AscendHostMemoryRegistrar:
    def __init__(self) -> None:
        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            raise RuntimeError("Ascend host registration support is unavailable")

        # Reuse the ACL runtime context owned by torch_npu/vLLM-Ascend. LMCache
        # must not call aclInit, aclFinalize, or aclrtResetDevice in-process.
        try:
            npu.current_device()
            npu.current_stream()
        except Exception as exc:
            raise RuntimeError("Cannot access the current Ascend context") from exc

        ascendcl = _load_ascendcl()
        try:
            self.host_register = ascendcl.aclrtHostRegisterV2
            self.host_unregister = ascendcl.aclrtHostUnregister
        except AttributeError as exc:
            raise RuntimeError(
                "The installed CANN runtime does not provide "
                "aclrtHostRegisterV2/aclrtHostUnregister"
            ) from exc

        self.host_register.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
        ]
        self.host_register.restype = ctypes.c_int
        self.host_unregister.argtypes = [ctypes.c_void_p]
        self.host_unregister.restype = ctypes.c_int

    def register(self, ptr: int, size: int) -> None:
        page_size = os.sysconf("SC_PAGESIZE")
        if ptr % page_size != 0 or size % page_size != 0:
            raise RuntimeError(
                "aclrtHostRegisterV2 requires a page-aligned address and size: "
                f"ptr=0x{ptr:x}, size={size}, page_size={page_size}"
            )
        flags = _ACL_HOST_REGISTER_TYPE | _ACL_HOST_REGISTER_MAP
        result = int(self.host_register(ctypes.c_void_p(ptr), size, flags))
        if result != _ACL_SUCCESS:
            raise RuntimeError(f"aclrtHostRegisterV2 returned error {result}")

    def unregister(self, ptr: int) -> None:
        result = int(self.host_unregister(ctypes.c_void_p(ptr)))
        if result != _ACL_SUCCESS:
            raise RuntimeError(f"aclrtHostUnregister returned error {result}")


def _ascend_is_available() -> bool:
    npu = getattr(torch, "npu", None)
    if npu is None:
        return False
    try:
        return bool(npu.is_available())
    except Exception:
        return False


def _create_host_memory_registrar() -> _HostMemoryRegistrar:
    if _ascend_is_available():
        return _AscendHostMemoryRegistrar()
    return _CudaHostMemoryRegistrar()


def create_mooncake_pinned_alloc_free(size: int) -> PinnedAllocFree:
    """Create callbacks for one Mooncake-backed LocalCPUBackend arena.

    Args:
        size: Number of bytes in the eager LocalCPUBackend arena.

    Returns:
        Allocation callbacks that allocate through Mooncake and register the
        resulting host range with the active CUDA or Ascend runtime.

    Raises:
        RuntimeError: If the Mooncake callbacks or accelerator host
            registration are unavailable, or allocation or registration fails.
    """
    try:
        # Third Party
        from mooncake.store import get_alloc_func_addr, get_free_func_addr
    except ImportError as exc:
        raise RuntimeError(
            "Mooncake Local CPU allocation requires mooncake.store with "
            "get_alloc_func_addr/get_free_func_addr"
        ) from exc

    alloc_addr = int(get_alloc_func_addr())
    free_addr = int(get_free_func_addr())
    if alloc_addr == 0 or free_addr == 0:
        raise RuntimeError(
            "Mooncake allocator callbacks are unavailable; build Mooncake "
            "with USE_NOF enabled"
        )

    registrar = _create_host_memory_registrar()
    alloc_callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
    free_callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    raw_alloc = alloc_callback_type(alloc_addr)
    raw_free = free_callback_type(free_addr)

    def alloc(requested_size: int) -> int:
        if requested_size != size:
            raise RuntimeError(
                "Mooncake allocator size changed after provider creation: "
                f"expected {size}, got {requested_size}"
            )
        ptr = int(raw_alloc(requested_size) or 0)
        if ptr == 0:
            raise RuntimeError(
                f"Mooncake allocator failed to allocate {requested_size} bytes"
            )
        try:
            registrar.register(ptr, requested_size)
        except Exception as exc:
            raw_free(ptr)
            raise RuntimeError(
                "Failed to register Mooncake memory with the accelerator runtime"
            ) from exc
        return ptr

    def free(ptr: int) -> None:
        unregister_exception: Exception | None = None
        try:
            registrar.unregister(ptr)
        except Exception as exc:
            unregister_exception = exc
        finally:
            raw_free(ptr)
        if unregister_exception is not None:
            raise RuntimeError(
                "Failed to unregister Mooncake memory from the accelerator runtime"
            ) from unregister_exception

    return PinnedAllocFree(
        alloc_fn=alloc,
        alloc_args=(),
        free_fn=free,
        free_args=(),
    )
