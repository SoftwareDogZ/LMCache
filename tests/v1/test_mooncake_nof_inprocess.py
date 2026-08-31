# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for in-process Mooncake NoF support."""

# Standard
from collections import OrderedDict
from types import ModuleType, SimpleNamespace
from typing import Any, cast
import asyncio
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.exceptions import IrrecoverableException
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager
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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_local_cpu_size": 0}, "max_local_cpu_size"),
        ({"local_cpu_use_hugepages": True}, "local_cpu_use_hugepages"),
        ({"enable_lazy_memory_allocator": True}, "enable_lazy_memory_allocator"),
        ({"enable_p2p": True}, "enable_p2p"),
        (
            {
                "extra_config": {
                    "save_chunk_meta": False,
                    "rust_raw_block.io_engine": "io_uring",
                }
            },
            "io_uring",
        ),
        (
            {
                "extra_config": {
                    "save_chunk_meta": False,
                    "local_cpu.pinned_align_bytes": 8192,
                }
            },
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
    monkeypatch.setattr(
        local_cpu_backend_module,
        "MixedMemoryAllocator",
        FakeAllocator,
    )

    config = _nof_config()
    config.validate()
    backend = LocalCPUBackend(config=config, metadata=_metadata())

    assert captured["provider_size"] == int(config.max_local_cpu_size * 1024**3)
    assert captured["pinned_alloc_free"] is callback_marker
    assert captured["align_bytes"] == 4096
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
    package = ModuleType("mooncake")
    store_module = ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = _FakeMooncakeStore  # type: ignore[attr-defined]
    store_module.ReplicateConfig = _FakeReplicateConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)


@pytest.mark.parametrize("save_chunk_meta", [False, True])
def test_connector_passes_configured_nof_count_to_store_paths(
    fake_mooncake: None,
    save_chunk_meta: bool,
) -> None:
    """Both zero-copy and metadata store paths preserve the configured count."""
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


def test_disabled_nof_switch_uses_zero_effective_replicas(
    fake_mooncake: None,
) -> None:
    """The switch disables NoF writes even when a positive count is configured."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=2,
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
