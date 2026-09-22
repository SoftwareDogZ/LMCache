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
            raise RuntimeError(
                "Ascend host registration support is unavailable"
            )

        # Reuse torch_npu/vLLM-Ascend's ACL context.
        try:
            npu.current_device()
            npu.current_stream()
        except Exception as exc:
            raise RuntimeError(
                "Cannot access the current Ascend context"
            ) from exc

        ascendcl = _load_ascendcl()

        try:
            self.host_register = ascendcl.aclrtHostRegisterV2
            self.host_get_device_pointer = (
                ascendcl.aclrtHostGetDevicePointer
            )
            self.host_unregister = ascendcl.aclrtHostUnregister
        except AttributeError as exc:
            raise RuntimeError(
                "The installed CANN runtime does not provide "
                "aclrtHostRegisterV2/"
                "aclrtHostGetDevicePointer/"
                "aclrtHostUnregister"
            ) from exc

        self.host_register.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
        ]
        self.host_register.restype = ctypes.c_int

        self.host_get_device_pointer.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
        ]
        self.host_get_device_pointer.restype = ctypes.c_int

        self.host_unregister.argtypes = [
            ctypes.c_void_p,
        ]
        self.host_unregister.restype = ctypes.c_int

    def register(self, ptr: int, size: int) -> None:
        page_size = os.sysconf("SC_PAGESIZE")

        if ptr % page_size != 0 or size % page_size != 0:
            raise RuntimeError(
                "aclrtHostRegisterV2 requires a page-aligned "
                "address and size: "
                f"ptr=0x{ptr:x}, size={size}, "
                f"page_size={page_size}"
            )

        flags = (
            _ACL_HOST_REGISTER_TYPE
            | _ACL_HOST_REGISTER_MAP
        )

        result = int(
            self.host_register(
                ctypes.c_void_p(ptr),
                size,
                flags,
            )
        )

        if result != _ACL_SUCCESS:
            raise RuntimeError(
                f"aclrtHostRegisterV2 returned error {result}"
            )

        # Get the NPU-visible address corresponding to this
        # registered Host address.
        dev_ptr = ctypes.c_void_p()

        result = int(
            self.host_get_device_pointer(
                ctypes.c_void_p(ptr),
                ctypes.byref(dev_ptr),
                0,
            )
        )

        if result != _ACL_SUCCESS:
            self.host_unregister(ctypes.c_void_p(ptr))
            raise RuntimeError(
                "aclrtHostGetDevicePointer returned error "
                f"{result}"
            )

        dev_addr = int(dev_ptr.value or 0)

        if dev_addr == 0:
            self.host_unregister(ctypes.c_void_p(ptr))
            raise RuntimeError(
                "aclrtHostGetDevicePointer returned NULL"
            )

        # IMPORTANT:
        # Import lazily. Importing lmcache_ascend at module load
        # time causes lmcache <-> lmcache_ascend circular import.
        from lmcache_ascend import c_ops as lmc_ops

        try:
            lmc_ops.register_mapping(
                int(ptr),
                int(dev_addr),
                int(size),
            )

            lookup = int(
                lmc_ops.get_device_ptr(int(ptr)) or 0
            )

            if lookup != dev_addr:
                raise RuntimeError(
                    "LMCache-Ascend mapping mismatch: "
                    f"host=0x{ptr:x}, "
                    f"dev=0x{dev_addr:x}, "
                    f"lookup=0x{lookup:x}"
                )

        except Exception:
            # If register_mapping succeeded, unregister_ptr()
            # also removes the manager record and ACL registration.
            try:
                lookup = int(
                    lmc_ops.get_device_ptr(int(ptr)) or 0
                )

                if lookup:
                    lmc_ops.unregister_ptr(int(ptr))
                else:
                    self.host_unregister(
                        ctypes.c_void_p(ptr)
                    )
            except Exception:
                self.host_unregister(
                    ctypes.c_void_p(ptr)
                )

            raise

        print(
            "[Mooncake][Ascend] mapping registered "
            f"host=0x{ptr:x} "
            f"dev=0x{dev_addr:x} "
            f"lookup=0x{lookup:x} "
            f"size={size}",
            flush=True,
        )

    def unregister(self, ptr: int) -> None:
        # register_mapping() added this pointer to
        # HostRegisteredMemoryManager, so unregister through
        # LMCache-Ascend as well. Its unregister_ptr() removes
        # the mapping and calls aclrtHostUnregister().
        from lmcache_ascend import c_ops as lmc_ops

        result = int(
            lmc_ops.unregister_ptr(int(ptr))
        )

        if result != _ACL_SUCCESS:
            raise RuntimeError(
                "LMCache-Ascend unregister_ptr returned "
                f"error {result}"
            )


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


def create_mooncake_pinned_alloc_free(
    size: int, allocator: str = "mooncake", numa_node: int = -1
) -> PinnedAllocFree:
    """Create callbacks for one Mooncake-backed LocalCPUBackend arena.

    Args:
        size: Number of bytes in the eager LocalCPUBackend arena.
        allocator: "mooncake" for legacy SPDK allocation,
            "mooncake_mmap_huge2m" for a single-policy HugeTLB mapping, or
            "mooncake_mmap_huge2m_numa" for a HugeTLB mapping split equally
            across the nodes in ``MC_NOF_HOST_NUMA_NODES``.
        numa_node: NUMA node for HugeTLB allocation; -1 preserves default policy
            or configured node order. For the segmented allocator, a configured
            node is rotated to the first region. Mooncake must bind before first
            touch and SPDK registration.

    Returns:
        Allocation callbacks that allocate through Mooncake and register the
        resulting host range with the active CUDA or Ascend runtime.

    Raises:
        ValueError: If size, allocator, or NUMA node is invalid.
        RuntimeError: If the Mooncake callbacks or accelerator host
            registration are unavailable, or allocation or registration fails.

    Notes:
        New Mooncake bindings must export the allocator-specific address getters
        used below. HugeTLB allocators return void* (*)(size_t, int32_t), and
        their free callbacks return void (*)(void*).
        Allocation must round the mapping up to 2 MiB, bind before first touch,
        and register the full mapping with SPDK before returning. Free must
        unregister from SPDK before munmap using the recorded mapping length.
        The caller must drain all accelerator and network I/O before free.
    """
    if size <= 0:
        raise ValueError("Arena size must be positive")
    if allocator not in (
        "mooncake",
        "mooncake_mmap_huge2m",
        "mooncake_mmap_huge2m_numa",
    ):
        raise ValueError("Unsupported Mooncake allocator: " + allocator)
    if numa_node < -1 or numa_node > 2**31 - 1:
        raise ValueError("numa_node must be -1 or a non-negative int32")
    external_hugepages = allocator in (
        "mooncake_mmap_huge2m",
        "mooncake_mmap_huge2m_numa",
    )
    segmented_numa = allocator == "mooncake_mmap_huge2m_numa"
    try:
        # Third Party
        import mooncake.store as store
    except ImportError as exc:
        raise RuntimeError(
            "Mooncake Local CPU allocation requires mooncake.store with "
            "get_alloc_func_addr/get_free_func_addr"
        ) from exc

    if segmented_numa:
        alloc_name = "get_mmap_huge2m_numa_alloc_func_addr"
        free_name = "get_mmap_huge2m_numa_free_func_addr"
    elif external_hugepages:
        alloc_name = "get_mmap_huge2m_alloc_func_addr"
        free_name = "get_mmap_huge2m_free_func_addr"
    else:
        alloc_name = "get_alloc_func_addr"
        free_name = "get_free_func_addr"
    try:
        alloc_addr = int(getattr(store, alloc_name)())
        free_addr = int(getattr(store, free_name)())
    except AttributeError as exc:
        raise RuntimeError(
            f"Allocator {allocator} requires Mooncake bindings "
            f"{alloc_name}/{free_name}; install a compatible Mooncake build"
        ) from exc
    if alloc_addr == 0 or free_addr == 0:
        raise RuntimeError(
            "Mooncake allocator callbacks are unavailable; build Mooncake "
            "with USE_NOF enabled"
        )

    registrar = _create_host_memory_registrar()
    # New ABI: void* alloc(size_t requested_size, int32_t numa_node).
    # Mooncake owns 2 MiB rounding and ptr->mapped_size tracking; free remains
    # void free(void*). No fallback to the legacy allocator is permitted.
    alloc_callback_type = (
        ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32)
        if external_hugepages
        else ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
    )
    free_callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    raw_alloc = alloc_callback_type(alloc_addr)
    raw_free = free_callback_type(free_addr)

    def alloc(requested_size: int) -> int:
        if requested_size != size:
            raise RuntimeError(
                "Mooncake allocator size changed after provider creation: "
                f"expected {size}, got {requested_size}"
            )
        ptr = int(
            (
                raw_alloc(requested_size, numa_node)
                if external_hugepages
                else raw_alloc(requested_size)
            )
            or 0
        )
        if ptr == 0:
            raise RuntimeError(
                f"Mooncake allocator failed to allocate {requested_size} bytes"
            )
        try:
            if external_hugepages and ptr % (2 * 1024 * 1024) != 0:
                raise RuntimeError("Mooncake HugeTLB pointer must be 2 MiB aligned")
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
