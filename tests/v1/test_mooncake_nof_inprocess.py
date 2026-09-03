# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for in-process Mooncake NoF support."""

# Standard
from collections import OrderedDict
from types import ModuleType, SimpleNamespace
from typing import Any, cast
import asyncio
import ctypes
import os
import sys
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.exceptions import IrrecoverableException
from lmcache.v1.memory_management import (
    MemoryFormat,
    MixedMemoryAllocator,
    PinnedAllocFree,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mooncake_memory_provider import create_mooncake_pinned_alloc_free
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager
import lmcache.v1.memory_management as memory_management_module
import lmcache.v1.mooncake_memory_provider as provider_module
import lmcache.v1.storage_backend.local_cpu_backend as local_cpu_backend_module


def _nof_config(
    replica_num: Any = 3,
    *,
    save_chunk_meta: bool = False,
    **overrides: Any,
) -> LMCacheEngineConfig:
    values: dict[str, Any] = {
        "chunk_size": 2,
        "enable_mooncake_nof_pool": True,
        "mooncake_nof_replica_num": replica_num,
        "max_local_cpu_size": 0.01,
        "remote_storage_plugins": ["mooncakestore"],
        "extra_config": {"save_chunk_meta": save_chunk_meta},
    }
    values.update(overrides)
    return LMCacheEngineConfig.from_defaults(**values)


def _metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float32,
        kv_shape=(1, 2, 2, 1, 1),
    )


@pytest.mark.parametrize("replica_num", [1, 2, 3, 17])
def test_nof_replica_num_accepts_positive_counts(replica_num: int) -> None:
    """Enabled NoF accepts any positive integer replica count."""
    config = _nof_config(replica_num)
    config.validate()
    assert config.mooncake_nof_replica_num == replica_num
    assert config.uses_mooncake_local_cpu_allocator() is True


@pytest.mark.parametrize("replica_num", [0, -1, True, 1.5])
def test_nof_replica_num_rejects_invalid_enabled_values(replica_num: Any) -> None:
    """Enabled NoF rejects zero, negative, boolean, and fractional counts."""
    config = _nof_config(replica_num)
    with pytest.raises(ValueError, match="mooncake_nof_replica_num"):
        config.validate()


def test_nof_replica_num_loads_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment configuration preserves replica counts greater than one."""
    monkeypatch.setenv("LMCACHE_ENABLE_MOONCAKE_NOF_POOL", "true")
    monkeypatch.setenv("LMCACHE_MOONCAKE_NOF_REPLICA_NUM", "8")
    monkeypatch.setenv("LMCACHE_REMOTE_STORAGE_PLUGINS", "mooncakestore")
    config = LMCacheEngineConfig.from_env()
    config.validate()
    assert config.mooncake_nof_replica_num == 8


def test_nof_requires_mooncake_remote_backend() -> None:
    """NoF allocation is rejected without an in-process Mooncake backend."""
    config = _nof_config(remote_storage_plugins=None)
    with pytest.raises(ValueError, match="requires a mooncakestore remote backend"):
        config.validate()


def test_mooncake_allocator_can_be_enabled_without_nof() -> None:
    """Mooncake allocation is independently selectable while NoF stays off."""
    config = LMCacheEngineConfig.from_defaults(
        local_cpu_allocator="mooncake",
        enable_mooncake_nof_pool=False,
        remote_storage_plugins=["mooncakestore"],
    )
    config.validate()

    assert config.uses_mooncake_local_cpu_allocator() is True
    assert config.enable_mooncake_nof_pool is False


def test_mooncake_allocator_rejects_unknown_value() -> None:
    """The public configuration rejects unknown Local CPU allocators."""
    config = LMCacheEngineConfig.from_defaults(local_cpu_allocator="unknown")
    with pytest.raises(ValueError, match="local_cpu_allocator"):
        config.validate()


def test_mooncake_allocator_constraints_apply_without_nof() -> None:
    """Allocator safety constraints do not depend on the NoF switch."""
    config = LMCacheEngineConfig.from_defaults(
        local_cpu_allocator="mooncake",
        enable_mooncake_nof_pool=False,
        max_local_cpu_size=0,
        remote_storage_plugins=["mooncakestore"],
    )
    with pytest.raises(ValueError, match="max_local_cpu_size"):
        config.validate()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_local_cpu_size": 0}, "max_local_cpu_size"),
        ({"enable_lazy_memory_allocator": True}, "enable_lazy_memory_allocator"),
        ({"enable_p2p": True}, "enable_p2p"),
        (
            {"extra_config": {"shm_name": "nof-test"}},
            "shm_name",
        ),
        (
            {"extra_config": {"rust_raw_block.io_engine": "io_uring"}},
            "io_uring",
        ),
        (
            {"extra_config": {"local_cpu.pinned_align_bytes": 8192}},
            "pinned_align_bytes",
        ),
    ],
)
def test_nof_rejects_incompatible_allocator_modes(
    overrides: dict[str, Any], message: str
) -> None:
    """NoF validation rejects modes that cannot share its contiguous arena."""
    config = _nof_config(**overrides)
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_nof_allows_independent_nixl_cpu_pool() -> None:
    """This version's independent NIXL CPU pool may coexist with NoF."""
    config = _nof_config(
        nixl_buffer_size=4096,
        nixl_buffer_device="cpu",
        extra_config={
            "enable_nixl_storage": True,
            "nixl_backend": "POSIX",
            "nixl_pool_size": 1,
        },
    )
    config.validate()


def test_mixed_allocator_uses_injected_callbacks() -> None:
    """The mixed allocator pairs an injected arena allocation and release."""
    backing = (ctypes.c_uint8 * 64)()
    calls: list[tuple[str, int]] = []

    def alloc(size: int) -> int:
        calls.append(("alloc", size))
        return ctypes.addressof(backing)

    def free(ptr: int) -> None:
        calls.append(("free", ptr))

    allocator = MixedMemoryAllocator(
        64,
        pinned_alloc_free=PinnedAllocFree(alloc, (), free, ()),
    )
    assert allocator.get_pinned_buffer().data_ptr() == ctypes.addressof(backing)
    allocator.close()
    assert calls == [("alloc", 64), ("free", ctypes.addressof(backing))]


def test_mixed_allocator_rolls_back_when_tensor_view_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arena view failures release a successfully registered custom buffer."""
    backing = (ctypes.c_uint8 * 64)()
    freed: list[int] = []

    callbacks = PinnedAllocFree(
        alloc_fn=lambda size: ctypes.addressof(backing),
        alloc_args=(),
        free_fn=lambda ptr: freed.append(ptr),
        free_args=(),
    )

    def fail_frombuffer(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("view failed")

    monkeypatch.setattr(
        memory_management_module.torch,
        "frombuffer",
        fail_frombuffer,
    )

    with pytest.raises(RuntimeError, match="view failed"):
        MixedMemoryAllocator(64, pinned_alloc_free=callbacks)

    assert freed == [ctypes.addressof(backing)]


def test_mooncake_provider_rolls_back_failed_cuda_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed CUDA registration releases the Mooncake allocation."""
    backing = (ctypes.c_uint8 * 64)()
    freed: list[int] = []
    alloc_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
    free_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    alloc_callback = alloc_type(lambda size: ctypes.addressof(backing))
    free_callback = free_type(lambda ptr: freed.append(int(ptr)))

    store_module = ModuleType("mooncake.store")
    setattr(
        store_module,
        "get_alloc_func_addr",
        lambda: ctypes.cast(alloc_callback, ctypes.c_void_p).value,
    )
    setattr(
        store_module,
        "get_free_func_addr",
        lambda: ctypes.cast(free_callback, ctypes.c_void_p).value,
    )
    monkeypatch.setitem(sys.modules, "mooncake", ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    monkeypatch.setattr(provider_module.torch_dev, "is_available", lambda: True)
    monkeypatch.setattr(
        provider_module.torch_dev,
        "cudart",
        lambda: SimpleNamespace(cudaHostRegister=lambda *args: 1),
    )

    callbacks = create_mooncake_pinned_alloc_free(64)
    with pytest.raises(RuntimeError, match="register Mooncake memory"):
        callbacks.alloc_fn(64)
    assert freed == [ctypes.addressof(backing)]


def test_mooncake_provider_pairs_registration_with_unregistration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful arena uses matching register, unregister, and free calls."""
    backing = (ctypes.c_uint8 * 64)()
    calls: list[tuple[str, int, int | None]] = []
    alloc_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
    free_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    alloc_callback = alloc_type(lambda size: ctypes.addressof(backing))
    free_callback = free_type(lambda ptr: calls.append(("free", int(ptr), None)))

    store_module = ModuleType("mooncake.store")
    setattr(
        store_module,
        "get_alloc_func_addr",
        lambda: ctypes.cast(alloc_callback, ctypes.c_void_p).value,
    )
    setattr(
        store_module,
        "get_free_func_addr",
        lambda: ctypes.cast(free_callback, ctypes.c_void_p).value,
    )
    monkeypatch.setitem(sys.modules, "mooncake", ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)

    registrar = SimpleNamespace(
        register=lambda ptr, size: calls.append(("register", ptr, size)),
        unregister=lambda ptr: calls.append(("unregister", ptr, None)),
    )
    monkeypatch.setattr(
        provider_module, "_create_host_memory_registrar", lambda: registrar
    )

    callbacks = create_mooncake_pinned_alloc_free(64)
    ptr = callbacks.alloc_fn(64)
    callbacks.free_fn(ptr)

    assert calls == [
        ("register", ctypes.addressof(backing), 64),
        ("unregister", ctypes.addressof(backing), None),
        ("free", ctypes.addressof(backing), None),
    ]


def test_mooncake_provider_uses_ascend_v2_dual_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Ascend provider maps an external allocation through ACL V2."""
    page_size = os.sysconf("SC_PAGESIZE")
    ptr = page_size * 1024
    freed: list[int] = []
    acl_calls: list[tuple[str, int, int | None, int | None]] = []
    alloc_type = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
    free_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    alloc_callback = alloc_type(lambda size: ptr)
    free_callback = free_type(lambda freed_ptr: freed.append(int(freed_ptr)))

    store_module = ModuleType("mooncake.store")
    setattr(
        store_module,
        "get_alloc_func_addr",
        lambda: ctypes.cast(alloc_callback, ctypes.c_void_p).value,
    )
    setattr(
        store_module,
        "get_free_func_addr",
        lambda: ctypes.cast(free_callback, ctypes.c_void_p).value,
    )
    monkeypatch.setitem(sys.modules, "mooncake", ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)

    class FakeAclFunction:
        def __init__(self, name: str) -> None:
            self.name = name
            self.argtypes: list[Any] = []
            self.restype: Any = None

        def __call__(self, *args: Any) -> int:
            acl_ptr = cast(ctypes.c_void_p, args[0]).value
            size = int(args[1]) if len(args) > 1 else None
            flags = int(args[2]) if len(args) > 2 else None
            acl_calls.append((self.name, int(acl_ptr or 0), size, flags))
            return 0

    fake_ascendcl = SimpleNamespace(
        aclrtHostRegisterV2=FakeAclFunction("register"),
        aclrtHostUnregister=FakeAclFunction("unregister"),
    )
    fake_npu = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        current_stream=lambda: object(),
    )
    monkeypatch.setattr(provider_module.torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(provider_module, "_load_ascendcl", lambda: fake_ascendcl)

    callbacks = create_mooncake_pinned_alloc_free(page_size)
    allocated_ptr = callbacks.alloc_fn(page_size)
    callbacks.free_fn(allocated_ptr)

    expected_flags = 0x10000002
    assert acl_calls == [
        ("register", ptr, page_size, expected_flags),
        ("unregister", ptr, None, None),
    ]
    assert fake_ascendcl.aclrtHostRegisterV2.argtypes == [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint32,
    ]
    assert freed == [ptr]


def test_local_cpu_backend_uses_mooncake_allocator_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public backend constructor injects Mooncake callbacks into its arena."""
    callback_marker = object()
    captured: dict[str, Any] = {}

    class FakeAllocator:
        def __init__(self, size: int, **kwargs: Any) -> None:
            captured["size"] = size
            captured.update(kwargs)
            self.buffer = torch.empty(16, dtype=torch.uint8)

        def get_pinned_buffer(self) -> torch.Tensor:
            return self.buffer

        def close(self) -> None:
            return None

    def create_provider(size: int) -> object:
        captured["provider_size"] = size
        return callback_marker

    monkeypatch.setattr(
        local_cpu_backend_module,
        "create_mooncake_pinned_alloc_free",
        create_provider,
    )
    monkeypatch.setattr(local_cpu_backend_module, "MixedMemoryAllocator", FakeAllocator)

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=2,
        max_local_cpu_size=0.01,
        local_cpu_allocator="mooncake",
        enable_mooncake_nof_pool=False,
        remote_storage_plugins=["mooncakestore"],
    )
    config.validate()
    backend = LocalCPUBackend(config=config, metadata=_metadata())

    configured_size = int(config.max_local_cpu_size * 1024**3)
    page_size = os.sysconf("SC_PAGESIZE")
    assert captured["provider_size"] == configured_size - configured_size % page_size
    assert captured["pinned_alloc_free"] is callback_marker
    assert captured["align_bytes"] == page_size
    assert backend.get_pinned_buffer() is not None
    backend.close()


class _FakeReplicateConfig:
    def __init__(self) -> None:
        self.replica_num = 0
        self.nof_replica_num = 0
        self.preferred_segment = ""


class _FakeMooncakeStore:
    instances: list["_FakeMooncakeStore"] = []
    registration_result = 0

    def __init__(self) -> None:
        self.closed = False
        self.calls: list[tuple[str, Any]] = []
        self.__class__.instances.append(self)

    def setup(self, *args: Any) -> None:
        self.calls.append(("setup", args))

    def get_hostname(self) -> str:
        return "test-host"

    def register_buffer(self, ptr: int, size: int) -> int:
        self.calls.append(("register_buffer", (ptr, size)))
        return self.registration_result

    def unregister_buffer(self, ptr: int) -> int:
        self.calls.append(("unregister_buffer", ptr))
        return 0

    def put_from(
        self, key: str, ptr: int, size: int, config: _FakeReplicateConfig
    ) -> int:
        self.calls.append(("put_from", config))
        return 0

    def batch_put_from(
        self,
        keys: list[str],
        ptrs: list[int],
        sizes: list[int],
        config: _FakeReplicateConfig,
    ) -> list[int]:
        self.calls.append(("batch_put_from", config))
        return [0 for _ in keys]

    def put_parts(
        self,
        key: str,
        *parts: bytes,
        config: _FakeReplicateConfig,
    ) -> int:
        self.calls.append(("put_parts", config))
        return 0

    def close(self) -> None:
        self.closed = True


class _FakeLocalCPUBackend:
    def __init__(self, config: LMCacheEngineConfig) -> None:
        self.config = config
        self.metadata = _metadata()
        self.buffer = torch.empty(64, dtype=torch.uint8)

    def get_memory_allocator(self) -> Any:
        return SimpleNamespace(numa_mapping=None)

    def get_pinned_buffer(self) -> torch.Tensor:
        return self.buffer


class _FakeMemoryObj:
    def __init__(self) -> None:
        self.raw_tensor = torch.zeros(8, dtype=torch.uint8)

    @property
    def data_ptr(self) -> int:
        return self.raw_tensor.data_ptr()

    @property
    def byte_array(self) -> bytes:
        return bytes(8)

    def get_size(self) -> int:
        return 8

    def get_shapes(self) -> list[torch.Size]:
        return [torch.Size([8])]

    def get_dtypes(self) -> list[torch.dtype]:
        return [torch.uint8]

    def get_memory_format(self) -> MemoryFormat:
        return MemoryFormat.KV_2LTD


@pytest.fixture
def fake_mooncake(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install an in-memory Mooncake binding used by connector tests."""
    _FakeMooncakeStore.instances.clear()
    _FakeMooncakeStore.registration_result = 0
    store_module = ModuleType("mooncake.store")
    setattr(store_module, "MooncakeDistributedStore", _FakeMooncakeStore)
    setattr(store_module, "ReplicateConfig", _FakeReplicateConfig)
    monkeypatch.setitem(sys.modules, "mooncake", ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)


@pytest.mark.parametrize("save_chunk_meta", [False, True])
def test_connector_passes_configured_nof_count_to_store_paths(
    fake_mooncake: None,
    save_chunk_meta: bool,
) -> None:
    """Every store path preserves the configured NoF replica count."""
    config = _nof_config(5, save_chunk_meta=save_chunk_meta)
    config.validate()
    loop = asyncio.new_event_loop()
    connector = MooncakestoreConnector(
        loop,
        _FakeLocalCPUBackend(config),  # type: ignore[arg-type]
        config,
    )
    key = CacheEngineKey("test_model", 1, 0, 1, torch.float32)
    memory_obj = _FakeMemoryObj()

    asyncio.run(connector.put(key, memory_obj))  # type: ignore[arg-type]
    asyncio.run(connector.batched_put([key], [memory_obj]))  # type: ignore[list-item]

    store_calls = _FakeMooncakeStore.instances[-1].calls
    operation_calls = [
        config_arg
        for operation, config_arg in store_calls
        if operation in {"put_from", "batch_put_from", "put_parts"}
    ]
    assert operation_calls
    assert all(call.nof_replica_num == 5 for call in operation_calls)
    assert all(call.replica_num == 1 for call in operation_calls)
    asyncio.run(connector.close())
    loop.close()


def test_connector_registration_failure_is_fatal_for_nof(
    fake_mooncake: None,
) -> None:
    """NoF startup fails and closes Mooncake when buffer registration fails."""
    _FakeMooncakeStore.registration_result = -1
    config = _nof_config()
    config.validate()
    loop = asyncio.new_event_loop()

    with pytest.raises(IrrecoverableException, match="registration failed"):
        MooncakestoreConnector(
            loop,
            _FakeLocalCPUBackend(config),  # type: ignore[arg-type]
            config,
        )

    assert _FakeMooncakeStore.instances[-1].closed is True
    loop.close()


def test_connector_registration_failure_is_fatal_for_mooncake_allocator(
    fake_mooncake: None,
) -> None:
    """Mooncake allocation without NoF still requires store registration."""
    _FakeMooncakeStore.registration_result = -1
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=2,
        local_cpu_allocator="mooncake",
        enable_mooncake_nof_pool=False,
        remote_storage_plugins=["mooncakestore"],
    )
    config.validate()
    loop = asyncio.new_event_loop()

    with pytest.raises(IrrecoverableException, match="registration failed"):
        MooncakestoreConnector(
            loop,
            _FakeLocalCPUBackend(config),  # type: ignore[arg-type]
            config,
        )

    assert _FakeMooncakeStore.instances[-1].closed is True
    loop.close()


def test_disabled_nof_switch_uses_zero_effective_replicas(
    fake_mooncake: None,
) -> None:
    """The switch disables NoF writes even when a positive count is configured."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=2,
        local_cpu_allocator="mooncake",
        enable_mooncake_nof_pool=False,
        mooncake_nof_replica_num=9,
        remote_storage_plugins=["mooncakestore"],
        extra_config={"save_chunk_meta": False},
    )
    config.validate()
    loop = asyncio.new_event_loop()
    connector = MooncakestoreConnector(
        loop,
        _FakeLocalCPUBackend(config),  # type: ignore[arg-type]
        config,
    )

    assert connector.replica_config.nof_replica_num == 0
    asyncio.run(connector.close())
    loop.close()


def test_storage_manager_closes_dependents_before_local_cpu() -> None:
    """Manager shutdown closes remote registrations before the local arena."""
    close_order: list[str] = []

    class FakeBackend:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            close_order.append(self.name)

    manager = StorageManager.__new__(StorageManager)
    manager.storage_backends = cast(
        Any,
        OrderedDict(
            [
                ("LocalCPUBackend", FakeBackend("local")),
                ("RemoteBackend-mooncakestore", FakeBackend("remote")),
            ]
        ),
    )
    manager.loop = cast(Any, SimpleNamespace(is_running=lambda: False))
    manager.thread = cast(Any, SimpleNamespace(is_alive=lambda: False))

    manager.close()

    assert close_order == ["remote", "local"]


def test_storage_manager_refuses_to_close_live_mooncake_arena() -> None:
    """Dynamic APIs cannot free the Mooncake arena before unregistration."""
    manager = StorageManager.__new__(StorageManager)
    manager.config = LMCacheEngineConfig.from_defaults(
        local_cpu_allocator="mooncake",
        remote_storage_plugins=["mooncakestore"],
    )
    manager.manager_lock = threading.Lock()
    manager.storage_backends = cast(
        Any,
        OrderedDict(
            [
                ("LocalCPUBackend", SimpleNamespace(close=lambda: None)),
                ("RemoteBackend-mooncakestore", SimpleNamespace(close=lambda: None)),
            ]
        ),
    )

    assert manager.close_backend("LocalCPUBackend") is False
    with pytest.raises(RuntimeError, match="Cannot recreate LocalCPUBackend"):
        manager.recreate_backend("LocalCPUBackend")
