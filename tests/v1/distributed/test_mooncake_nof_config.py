# SPDX-License-Identifier: Apache-2.0
"""Tests for MP Mooncake NoF L1 and replica configuration."""

# Standard
from typing import Any, cast
import argparse
import ctypes
import sys
import types

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    L1MemoryManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.memory_manager import l1_memory_manager
from lmcache.v1.memory_allocators import mooncake_memory_provider
from lmcache.v1.multiprocess.config import add_mp_server_args


def _parse_mp_storage_args(args: list[str]) -> L1MemoryManagerConfig:
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    config = parse_args_to_config(parser.parse_args(args))
    return config.l1_manager_config.memory_config


def test_mooncake_nof_replica_num_parses_from_mp_cli() -> None:
    config = _parse_mp_storage_args(
        [
            "--l1-size-gb",
            "1",
            "--no-l1-use-lazy",
            "--eviction-policy",
            "LRU",
            "--enable-mooncake-nof-pool",
            "--mooncake-nof-replica-num",
            "3",
        ]
    )

    assert config.enable_mooncake_nof_pool is True
    assert config.mooncake_nof_replica_num == 3
    assert config.shm_name == ""


@pytest.mark.parametrize("replica_num", [-1, -2])
def test_mooncake_nof_replica_num_rejects_negative_values(
    replica_num: int,
) -> None:
    with pytest.raises(ValueError, match="mooncake_nof_replica_num must be >= 0"):
        L1MemoryManagerConfig(
            size_in_bytes=4096,
            use_lazy=False,
            shm_name="",
            mooncake_nof_replica_num=replica_num,
        )


def test_enabled_mooncake_nof_pool_requires_positive_replica_count() -> None:
    with pytest.raises(ValueError, match="mooncake_nof_replica_num to be >= 1"):
        L1MemoryManagerConfig(
            size_in_bytes=4096,
            use_lazy=False,
            shm_name="",
            enable_mooncake_nof_pool=True,
            mooncake_nof_replica_num=0,
        )


@pytest.mark.parametrize(
    ("size_in_bytes", "align_bytes", "match"),
    [
        (4096, 8192, "l1-align-bytes to be 4096"),
        (4097, 4096, "L1 size to be 4096-byte aligned"),
    ],
)
def test_enabled_mooncake_nof_pool_requires_4k_alignment(
    size_in_bytes: int,
    align_bytes: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        L1MemoryManagerConfig(
            size_in_bytes=size_in_bytes,
            use_lazy=False,
            align_bytes=align_bytes,
            shm_name="",
            enable_mooncake_nof_pool=True,
        )


@pytest.mark.parametrize(
    ("use_lazy", "shm_name", "match"),
    [
        (True, "", "requires lazy allocation to be disabled"),
        (False, "lmcache_l1_pool_test", "cannot be used with POSIX SHM"),
    ],
)
def test_enabled_mooncake_nof_pool_rejects_incompatible_l1_modes(
    use_lazy: bool,
    shm_name: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        L1MemoryManagerConfig(
            size_in_bytes=4096,
            use_lazy=use_lazy,
            shm_name=shm_name,
            enable_mooncake_nof_pool=True,
        )


@pytest.mark.parametrize(
    ("enabled", "configured_replica_num", "effective_replica_num"),
    [(False, 4, 0), (True, 4, 4)],
)
def test_l1_descriptor_reports_effective_mooncake_nof_replica_num(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    configured_replica_num: int,
    effective_replica_num: int,
) -> None:
    class FakeBuffer:
        def data_ptr(self) -> int:
            return 0x1000

    class FakeMixedMemoryAllocator(l1_memory_manager.MixedMemoryAllocator):
        def __init__(self) -> None:
            self.buffer = FakeBuffer()

        def close(self) -> None:
            return None

    fake_allocator = FakeMixedMemoryAllocator()

    monkeypatch.setattr(
        l1_memory_manager,
        "create_memory_allocator",
        lambda _config: fake_allocator,
    )
    config = L1MemoryManagerConfig(
        size_in_bytes=4096,
        use_lazy=False,
        shm_name="",
        enable_mooncake_nof_pool=enabled,
        mooncake_nof_replica_num=configured_replica_num,
    )

    manager = l1_memory_manager.L1MemoryManager(config)
    descriptor = manager.get_l1_memory_desc()
    manager.close()

    assert descriptor.mooncake_nof_replica_num == effective_replica_num


class _FakeDeviceSpec:
    def __init__(self, pin_result: bool = True) -> None:
        self.pin_result = pin_result
        self.pin_calls: list[tuple[int, int]] = []
        self.unpin_calls: list[int] = []

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        del flags
        self.pin_calls.append((ptr, size))
        return self.pin_result

    def unpin_memory(self, ptr: int) -> bool:
        self.unpin_calls.append(ptr)
        return True


def _install_fake_mooncake_store(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, list[int]]:
    backing = (ctypes.c_ubyte * 4096)()
    ptr = ctypes.addressof(backing)
    free_calls: list[int] = []

    alloc_callback_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
    free_callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

    def alloc(_size: int) -> int:
        return ptr

    def free(freed_ptr: int) -> None:
        free_calls.append(int(freed_ptr))

    alloc_callback = alloc_callback_type(alloc)
    free_callback = free_callback_type(free)

    store_module = types.ModuleType("mooncake.store")
    store_module_any = cast(Any, store_module)
    store_module_any.get_alloc_func_addr = lambda: ctypes.cast(
        alloc_callback, ctypes.c_void_p
    ).value
    store_module_any.get_free_func_addr = lambda: ctypes.cast(
        free_callback, ctypes.c_void_p
    ).value
    # Keep callbacks and backing storage alive for the duration of the test.
    store_module_any._test_refs = (backing, alloc_callback, free_callback)

    mooncake_module = types.ModuleType("mooncake")
    mooncake_module_any = cast(Any, mooncake_module)
    mooncake_module_any.store = store_module
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    return ptr, free_calls


def test_mooncake_memory_provider_pins_and_frees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ptr, free_calls = _install_fake_mooncake_store(monkeypatch)
    device = _FakeDeviceSpec()
    monkeypatch.setattr(mooncake_memory_provider, "current_device_spec", device)

    provider = mooncake_memory_provider.create_mooncake_pinned_alloc_free(4096)
    allocated_ptr = provider.alloc()
    provider.free(allocated_ptr)

    assert allocated_ptr == ptr
    assert device.pin_calls == [(ptr, 4096)]
    assert device.unpin_calls == [ptr]
    assert free_calls == [ptr]


def test_mooncake_memory_provider_rolls_back_when_pinning_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ptr, free_calls = _install_fake_mooncake_store(monkeypatch)
    device = _FakeDeviceSpec(pin_result=False)
    monkeypatch.setattr(mooncake_memory_provider, "current_device_spec", device)

    provider = mooncake_memory_provider.create_mooncake_pinned_alloc_free(4096)
    with pytest.raises(RuntimeError, match="register Mooncake NoF memory"):
        provider.alloc()

    assert device.pin_calls == [(ptr, 4096)]
    assert free_calls == [ptr]
