# SPDX-License-Identifier: Apache-2.0
"""Contract tests for external HugeTLB callbacks; no DMA hardware is required."""

# Standard
from types import ModuleType, SimpleNamespace
import ctypes
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.mooncake_memory_provider import create_mooncake_pinned_alloc_free
import lmcache.v1.mooncake_memory_provider as provider


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, list]:
    """Install fake Mooncake callbacks and record accelerator registrations."""
    calls: list = []
    store = ModuleType("mooncake.store")
    package = ModuleType("mooncake")
    package.store = store
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.store", store)
    monkeypatch.setattr(provider.torch, "npu", None, raising=False)
    monkeypatch.setattr(
        provider,
        "torch_dev",
        SimpleNamespace(
            is_available=lambda: True,
            cudart=lambda: SimpleNamespace(
                cudaHostRegister=lambda ptr, size, flags: (
                    calls.append(("register", ptr, size)) or 0
                ),
                cudaHostUnregister=lambda ptr: (calls.append(("unregister", ptr)) or 0),
            ),
        ),
    )
    return store, calls


def test_missing_new_binding_is_explicit(runtime: tuple) -> None:
    """New allocation never silently falls back to spdk_zmalloc."""
    store, _ = runtime
    store.get_alloc_func_addr = lambda: 1
    store.get_free_func_addr = lambda: 1
    with pytest.raises(RuntimeError, match="get_mmap_huge2m_alloc_func_addr"):
        create_mooncake_pinned_alloc_free(4096, "mooncake_mmap_huge2m")


@pytest.mark.parametrize("node", [-1, 0, 3])
def test_external_allocation_lifecycle(runtime: tuple, node: int) -> None:
    """Requested length and NUMA policy reach Mooncake; CUDA unregisters first."""
    store, calls = runtime
    address = 2 * 1024 * 1024
    alloc = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32)(
        lambda size, numa: calls.append(("alloc", size, numa)) or address
    )
    free = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
        lambda ptr: calls.append(("free", ptr))
    )
    store.get_mmap_huge2m_alloc_func_addr = lambda: ctypes.cast(
        alloc, ctypes.c_void_p
    ).value
    store.get_mmap_huge2m_free_func_addr = lambda: ctypes.cast(
        free, ctypes.c_void_p
    ).value
    callbacks = create_mooncake_pinned_alloc_free(4096, "mooncake_mmap_huge2m", node)
    ptr = callbacks.alloc_fn(4096)
    callbacks.free_fn(ptr)
    assert calls == [
        ("alloc", 4096, node),
        ("register", address, 4096),
        ("unregister", address),
        ("free", address),
    ]


@pytest.mark.parametrize("node", [-1, 0, 3])
def test_segmented_numa_allocation_lifecycle(runtime: tuple, node: int) -> None:
    """The segmented mode selects its own Mooncake callbacks and forwards preference."""
    store, calls = runtime
    address = 2 * 1024 * 1024
    alloc = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32)(
        lambda size, numa: calls.append(("numa_alloc", size, numa)) or address
    )
    free = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
        lambda ptr: calls.append(("numa_free", ptr))
    )
    store.get_mmap_huge2m_numa_alloc_func_addr = lambda: ctypes.cast(
        alloc, ctypes.c_void_p
    ).value
    store.get_mmap_huge2m_numa_free_func_addr = lambda: ctypes.cast(
        free, ctypes.c_void_p
    ).value

    callbacks = create_mooncake_pinned_alloc_free(
        4096, "mooncake_mmap_huge2m_numa", node
    )
    ptr = callbacks.alloc_fn(4096)
    callbacks.free_fn(ptr)

    assert calls == [
        ("numa_alloc", 4096, node),
        ("register", address, 4096),
        ("unregister", address),
        ("numa_free", address),
    ]


def test_misaligned_allocation_is_released(runtime: tuple) -> None:
    """An invalid Mooncake pointer is freed without accelerator registration."""
    store, calls = runtime
    alloc = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32)(
        lambda size, numa: 4096
    )
    free = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
        lambda ptr: calls.append(("free", ptr))
    )
    store.get_mmap_huge2m_alloc_func_addr = lambda: ctypes.cast(
        alloc, ctypes.c_void_p
    ).value
    store.get_mmap_huge2m_free_func_addr = lambda: ctypes.cast(
        free, ctypes.c_void_p
    ).value
    callbacks = create_mooncake_pinned_alloc_free(4096, "mooncake_mmap_huge2m")
    with pytest.raises(RuntimeError, match="register Mooncake memory"):
        callbacks.alloc_fn(4096)
    assert calls == [("free", 4096)]


@pytest.mark.parametrize("register_result", [0, 1])
def test_ascend_registration(
    runtime: tuple, monkeypatch: pytest.MonkeyPatch, register_result: int
) -> None:
    """Ascend uses V2 host registration and preserves an existing NPU context."""
    store, calls = runtime
    address = 2 * 1024 * 1024
    alloc = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32)(
        lambda size, numa: address
    )
    free = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
        lambda ptr: calls.append(("free", ptr))
    )
    store.get_mmap_huge2m_alloc_func_addr = lambda: ctypes.cast(
        alloc, ctypes.c_void_p
    ).value
    store.get_mmap_huge2m_free_func_addr = lambda: ctypes.cast(
        free, ctypes.c_void_p
    ).value

    def register(ptr: ctypes.c_void_p, size: int, flags: int) -> int:
        calls.append(("acl_register", ptr.value, size, flags))
        return register_result

    def unregister(ptr: ctypes.c_void_p) -> int:
        calls.append(("acl_unregister", ptr.value))
        return 0

    monkeypatch.setattr(
        provider.torch,
        "npu",
        SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            current_stream=lambda: object(),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        provider.ctypes,
        "CDLL",
        lambda path: SimpleNamespace(
            aclrtHostRegisterV2=register, aclrtHostUnregister=unregister
        ),
    )
    callbacks = create_mooncake_pinned_alloc_free(4096, "mooncake_mmap_huge2m")
    if register_result:
        with pytest.raises(RuntimeError, match="register Mooncake memory"):
            callbacks.alloc_fn(4096)
        assert calls == [
            ("acl_register", address, 4096, 0x10000002),
            ("free", address),
        ]
        return
    ptr = callbacks.alloc_fn(4096)
    callbacks.free_fn(ptr)
    assert calls == [
        ("acl_register", address, 4096, 0x10000002),
        ("acl_unregister", address),
        ("free", address),
    ]
