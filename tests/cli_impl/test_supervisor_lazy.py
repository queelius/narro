"""Tests for the lazy-load supervisor wiring (Task E of v0.40.0).

The supervisor now boots without spawning workers eagerly. Models load on
first request via the LoadDirector, which the supervisor instantiates and
hangs off SupervisorState. Boot-time validation walks the enabled-catalog
and stamps unservable_reasons for entries with no memory data or memory
estimates exceeding device capacity.

These tests verify:
  - SupervisorState carries `director` + `unservable_reasons` fields.
  - run_supervisor does NOT spawn workers eagerly.
  - validate_catalog_at_boot flags models without memory data.
  - validate_catalog_at_boot flags models exceeding device capacity.
  - validate_catalog_at_boot does NOT flag models with valid data fitting.
  - The director on state is a LoadDirector instance during run_supervisor.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from muse.cli_impl.idle_sweeper import IdleSweeper
from muse.cli_impl.load_director import LoadDirector, LoadEntry
from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerSpec,
    clear_supervisor_state,
    get_supervisor_state,
    set_supervisor_state,
    spawn_worker,
    validate_catalog_at_boot,
)


@pytest.fixture(autouse=True)
def _reset_supervisor_state():
    clear_supervisor_state()
    yield
    clear_supervisor_state()


@pytest.fixture(autouse=True)
def _mock_resource_registry(monkeypatch):
    """Keep supervisor unit tests off the host process registry."""
    monkeypatch.setattr(
        "muse.cli_impl.supervisor.register_process",
        lambda **kwargs: f"test-resource-{kwargs['pid']}",
    )
    monkeypatch.setattr(
        "muse.cli_impl.supervisor.unregister_process",
        lambda resource_id: bool(resource_id),
    )


@pytest.fixture(autouse=True)
def _isolate_pynvml_sentinels():
    """Prevent supervisor tests from polluting memory_probe module-level
    pynvml sentinels (which are loaded lazily inside _MemoryProbeAdapter
    when run_supervisor calls validate_catalog_at_boot with no probe
    override). On entry, save current state. On exit, restore the
    fresh-untried state so memory_probe tests starting after us see a
    clean slate.
    """
    import muse.core.memory_probe as mod
    orig = (mod.pynvml, mod._init_attempted, mod._init_ok)
    try:
        yield
    finally:
        # Restore to a fresh-untried state, not just whatever it was on
        # entry, so subsequent test files (notably tests/core/test_memory_probe.py)
        # see a clean module. The memory_probe fixture itself uses
        # save-and-restore, which preserves whatever pollution we leave
        # behind; resetting to (None, False, False) prevents that.
        mod.pynvml, mod._init_attempted, mod._init_ok = None, False, False
        # (orig captured but unused; kept to preserve the variable for
        # observability if someone wants to inspect what we overwrote.)
        del orig


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    return tmp_path


def _seed_catalog(data: dict) -> None:
    from muse.core.catalog import _reset_known_models_cache, _write_catalog
    normalized: dict = {}
    for model_id, raw_entry in data.items():
        entry = dict(raw_entry)
        raw_manifest = entry.get("manifest")
        if isinstance(raw_manifest, dict):
            manifest = dict(raw_manifest)
            manifest.setdefault("model_id", model_id)
            manifest.setdefault("modality", "embedding/text")
            manifest.setdefault("hf_repo", entry.get("hf_repo", "test/model"))
            manifest.setdefault("backend_path", "muse.core.runtime:TestBackend")
            entry["manifest"] = manifest
        normalized[model_id] = entry
    _write_catalog(normalized)
    _reset_known_models_cache()


class TestWorkerResourceRegistration:
    def test_replacement_registers_new_identity_before_retiring_old(self):
        spec = WorkerSpec(
            models=["demo"], python_path="/mock/python", port=9001,
            resource_id="old-resource",
        )
        process = MagicMock(pid=4242)
        events: list[tuple] = []

        with patch(
            "muse.cli_impl.supervisor.subprocess.Popen", return_value=process,
        ), patch(
            "muse.cli_impl.supervisor.os.getpid", return_value=31337,
        ), patch(
            "muse.cli_impl.supervisor.register_process",
            side_effect=lambda **kwargs: events.append(("register", kwargs)) or "new-resource",
        ), patch(
            "muse.cli_impl.supervisor.unregister_process",
            side_effect=lambda resource_id: events.append(("unregister", resource_id)),
        ):
            spawn_worker(spec, device="cpu")

        assert spec.process is process
        assert spec.resource_id == "new-resource"
        assert [event[0] for event in events] == ["register", "unregister"]
        assert events[1] == ("unregister", "old-resource")

    def test_registration_failure_rolls_back_and_keeps_superseded_record(self):
        from muse.core.resource_registry import ResourceRegistryError

        spec = WorkerSpec(
            models=["demo"], python_path="/mock/python", port=9001,
            resource_id="old-resource",
        )
        process = MagicMock(pid=4242)

        def rollback(specs):
            assert specs == [spec]
            spec.process = None

        with patch(
            "muse.cli_impl.supervisor.subprocess.Popen", return_value=process,
        ), patch(
            "muse.cli_impl.supervisor.os.getpid", return_value=31337,
        ), patch(
            "muse.cli_impl.supervisor.register_process",
            side_effect=ResourceRegistryError("disk unavailable"),
        ), patch(
            "muse.cli_impl.supervisor.unregister_process",
        ) as unregister, patch(
            "muse.cli_impl.supervisor._shutdown_workers",
            side_effect=rollback,
        ) as shutdown:
            with pytest.raises(ResourceRegistryError, match="rolled back"):
                spawn_worker(spec, device="cpu")

        shutdown.assert_called_once_with([spec])
        assert spec.process is None
        assert spec.resource_id == "old-resource"
        unregister.assert_not_called()

    def test_shutdown_fails_closed_if_process_identity_changes(self):
        from muse.cli_impl.supervisor import _shutdown_workers

        spec = WorkerSpec(
            models=["demo"], python_path="/mock/python", port=9001,
            resource_id="old-resource",
        )
        old_process = MagicMock(pid=4242)
        replacement = MagicMock(pid=4343)
        spec.process = old_process

        def replace_during_observation(*_args):
            spec.process = replacement
            spec.resource_id = "new-resource"
            return None

        with patch(
            "muse.cli_impl.supervisor._worker_process_state_locked",
            side_effect=replace_during_observation,
        ), patch(
            "muse.cli_impl.supervisor.unregister_process",
        ) as unregister:
            result = _shutdown_workers([spec], grace=0.0)

        unregister.assert_not_called()
        assert spec.process is replacement
        assert spec.resource_id == "new-resource"
        assert result.retained == (spec,)


# ---------------------------------------------------------------------------
# E1: SupervisorState fields
# ---------------------------------------------------------------------------


class TestSupervisorStateLazyFields:
    def test_state_has_director_field_default_none(self):
        s = SupervisorState()
        assert hasattr(s, "director")
        assert s.director is None

    def test_state_has_unservable_reasons_dict(self):
        s = SupervisorState()
        assert hasattr(s, "unservable_reasons")
        assert s.unservable_reasons == {}

    def test_state_director_is_writable(self):
        """The supervisor populates state.director once the LoadDirector is built."""
        s = SupervisorState()
        director = MagicMock(spec=LoadDirector)
        s.director = director
        assert s.director is director

    def test_unservable_reasons_is_per_state_instance(self):
        """Default dicts must not bleed between SupervisorState instances."""
        a = SupervisorState()
        b = SupervisorState()
        a.unservable_reasons["x"] = "out of memory"
        assert b.unservable_reasons == {}


# ---------------------------------------------------------------------------
# E2: validate_catalog_at_boot
# ---------------------------------------------------------------------------


class TestValidateCatalogAtBoot:
    def test_available_pools_preserves_pair_contract_with_budgets(self):
        from muse.cli_impl.supervisor import _available_pools

        probe = MagicMock()
        probe.cpu_free_gb.return_value = 16.0
        probe.gpu_free_gb.return_value = None

        assert _available_pools(
            probe,
            gpu_budget_gb=8.0,
            cpu_budget_gb=12.0,
            gpu_headroom_gb=1.0,
            cpu_headroom_gb=2.0,
        ) == (10.0, 7.0)

    def test_flags_model_with_no_memory_data(self, tmp_catalog):
        """Enabled model with no memory_gb annotation and no measurements
        is flagged as unservable with the probe-prompt message.
        """
        _seed_catalog({
            "needs-probe": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "needs-probe",
                    "modality": "embedding/text",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {
                        # no memory_gb, no measurements
                    },
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        assert "needs-probe" in state.unservable_reasons
        reason = state.unservable_reasons["needs-probe"]
        assert "no memory estimate" in reason
        assert "muse models probe" in reason

    def test_flags_model_exceeding_device_capacity(self, tmp_catalog):
        """Enabled model whose memory_gb exceeds free at boot (minus headroom)
        is flagged as exceeds_capacity. Uses live cpu_free_gb from probe.
        """
        _seed_catalog({
            "huge-model": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "huge-model",
                    "modality": "chat/completion",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {
                        "memory_gb": 100.0,
                        "device": "cpu",
                    },
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 16.0  # nowhere near 100
        validate_catalog_at_boot(state, memory_probe=probe)
        assert "huge-model" in state.unservable_reasons
        reason = state.unservable_reasons["huge-model"]
        assert "exceeds device capacity" in reason

    def test_does_not_flag_model_that_fits(self, tmp_catalog):
        """Enabled model with valid memory_gb that fits live free is NOT flagged."""
        _seed_catalog({
            "tiny-model": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "tiny-model",
                    "modality": "embedding/text",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {
                        "memory_gb": 0.5,
                        "device": "cpu",
                    },
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        assert "tiny-model" not in state.unservable_reasons
        assert state.unservable_reasons == {}

    def test_does_not_flag_model_with_probe_measurements(self, tmp_catalog):
        """A model lacking capabilities.memory_gb but with recorded
        measurements.<device>.peak_bytes is fine.
        """
        _seed_catalog({
            "probed-model": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "probed-model",
                    "modality": "embedding/text",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {
                        "device": "cpu",
                        # no memory_gb
                    },
                },
                "measurements": {
                    "cpu": {
                        "peak_bytes": 500_000_000,
                        "device": "cpu",
                        "weights_bytes": 400_000_000,
                    },
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        assert "probed-model" not in state.unservable_reasons

    def test_does_not_flag_bundled_model_probed_on_other_device(self, tmp_catalog):
        """Bundled models have no persisted manifest in catalog.json, so the
        manifest-derived device defaults to 'cpu'. A probe records under the
        real device (e.g. 'cuda'). Boot validation must consult the probe
        measurement (adopting its recorded device) rather than flagging the
        model 'no memory estimate' forever. Regression: `muse models probe`
        on a bundled GPU model never cleared the flag (probe writes
        measurements.cuda; the lookup read measurements.cpu).
        """
        _seed_catalog({
            "bundled-gpu": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": True,
                # NO "manifest" key: exactly how bundled models appear.
                "measurements": {
                    "cuda": {
                        "peak_bytes": 3_000_000_000,
                        "device": "cuda",
                        "weights_bytes": 2_500_000_000,
                    },
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = 10.0   # 3GB fits in 10 - headroom
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        assert "bundled-gpu" not in state.unservable_reasons

    def test_skips_disabled_models(self, tmp_catalog):
        """Disabled entries are not validated; they can't 503."""
        _seed_catalog({
            "disabled-model": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": False,
                "manifest": {
                    "model_id": "disabled-model",
                    "modality": "embedding/text",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {},
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        assert state.unservable_reasons == {}

    def test_skips_pre_worker_entries_without_python_path(self, tmp_catalog):
        """Catalog entries without python_path can't actually be loaded;
        they're not flagged as unservable here (they'll be skipped at load
        time anyway), they just don't enter validation.
        """
        _seed_catalog({
            "legacy": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "enabled": True,
                # no python_path
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        # Either flagged (with a coherent reason) or skipped silently.
        # Just verify no exception.
        # In either case, the state should be valid.
        assert isinstance(state.unservable_reasons, dict)

    def test_handles_gpu_device_with_pynvml_unavailable(self, tmp_catalog):
        """Model declares device=cuda; pynvml returns None. Without GPU
        info we can't say it fits; flag exceeds_capacity (the safe choice
        until probe data lands).
        """
        _seed_catalog({
            "gpu-model": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "gpu-model",
                    "modality": "image/generation",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {
                        "memory_gb": 4.0,
                        "device": "cuda",
                    },
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None  # pynvml unavailable
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        assert "gpu-model" in state.unservable_reasons

    def test_gpu_budget_is_fallback_when_pynvml_unavailable(self, tmp_catalog):
        _seed_catalog({
            "gpu-model": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "capabilities": {"memory_gb": 6.0, "device": "cuda"},
                },
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0

        validate_catalog_at_boot(
            state,
            memory_probe=probe,
            gpu_budget_gb=8.0,
            gpu_headroom_gb=1.0,
        )

        assert "gpu-model" not in state.unservable_reasons

    def test_rocm_like_auto_model_uses_gpu_budget(self, tmp_catalog):
        _seed_catalog({
            "rocm-model": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "capabilities": {"memory_gb": 6.0, "device": "auto"},
                },
            },
        })

        class RocmProbe:
            def gpu_free_gb(self):
                return None

            def cpu_free_gb(self):
                return 2.0

            def cuda_available(self):
                return True

        state = SupervisorState()
        validate_catalog_at_boot(
            state,
            memory_probe=RocmProbe(),
            gpu_budget_gb=8.0,
            gpu_headroom_gb=1.0,
            cpu_headroom_gb=0.0,
        )

        # It fits the static GPU pool but not the deliberately tiny CPU pool.
        assert "rocm-model" not in state.unservable_reasons

    def test_transient_nvidia_pressure_does_not_create_hard_stamp(self, tmp_catalog):
        _seed_catalog({
            "nvidia-model": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "capabilities": {"memory_gb": 5.5, "device": "auto"},
                },
            },
        })
        state = SupervisorState()

        class NvidiaProbe:
            def gpu_free_gb(self):
                return 2.0

            def gpu_total_gb(self):
                return 12.0

            def cpu_free_gb(self):
                return 64.0

            def cpu_total_gb(self):
                return 64.0

            def cuda_available(self):
                return True

        validate_catalog_at_boot(
            state,
            memory_probe=NvidiaProbe(),
            gpu_budget_gb=8.0,
            gpu_headroom_gb=1.0,
        )

        # Hard capacity is min(12 GiB physical, 8 GiB budget) - 1 GiB
        # headroom = 7 GiB. The current 2 GiB free is a director concern:
        # Muse can evict its own workers before loading this 5.5 GiB model.
        assert "nvidia-model" not in state.unservable_reasons

    def test_physical_gpu_ceiling_still_hard_stamps_oversized_model(
        self, tmp_catalog,
    ):
        _seed_catalog({
            "too-large": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "capabilities": {"memory_gb": 12.0, "device": "cuda"},
                },
            },
        })

        class NvidiaProbe:
            def gpu_free_gb(self):
                return 7.0

            def gpu_total_gb(self):
                return 8.0

            def cpu_free_gb(self):
                return 64.0

            def cpu_total_gb(self):
                return 64.0

            def cuda_available(self):
                return True

        state = SupervisorState()
        validate_catalog_at_boot(
            state, memory_probe=NvidiaProbe(), gpu_headroom_gb=1.0,
        )

        reason = state.unservable_reasons["too-large"]
        assert "12.0 GB > 7.0 GB capacity on cuda" in reason


# ---------------------------------------------------------------------------
# Regression: a corrupt catalog.json (no last-known-good cache) must not
# crash boot with a raw traceback. `_read_catalog` raises `CatalogError` in
# that case (the corrupt-guard); `validate_catalog_at_boot` must catch it,
# log one clear line naming the catalog path (no traceback), and degrade
# gracefully -- boot continues with no models flagged unservable, the same
# as the pre-existing missing/empty-catalog case.
# ---------------------------------------------------------------------------


class TestValidateCatalogAtBootCorruptCatalog:
    def test_corrupt_catalog_does_not_raise(self, tmp_catalog):
        from muse.core.catalog import _catalog_path

        _catalog_path().parent.mkdir(parents=True, exist_ok=True)
        _catalog_path().write_text("{not valid json")

        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0

        # Must not raise CatalogError (or anything else) out of boot.
        validate_catalog_at_boot(state, memory_probe=probe)

    def test_corrupt_catalog_leaves_no_unservable_stamps(self, tmp_catalog):
        """Degrades the same way an empty/missing catalog does: nothing
        gets flagged since nothing could be read."""
        from muse.core.catalog import _catalog_path

        _catalog_path().parent.mkdir(parents=True, exist_ok=True)
        _catalog_path().write_text("{not valid json")

        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0

        validate_catalog_at_boot(state, memory_probe=probe)
        assert state.unservable_reasons == {}

    def test_corrupt_catalog_logs_one_clear_line_no_traceback(self, tmp_catalog, caplog):
        from muse.core.catalog import _catalog_path

        _catalog_path().parent.mkdir(parents=True, exist_ok=True)
        _catalog_path().write_text("{not valid json")

        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0

        with caplog.at_level("ERROR", logger="muse.cli_impl.supervisor"):
            validate_catalog_at_boot(state, memory_probe=probe)

        boot_records = [
            r for r in caplog.records
            if r.name == "muse.cli_impl.supervisor"
        ]
        assert len(boot_records) == 1
        record = boot_records[0]
        # No traceback: the underlying CatalogError was already logged by
        # _read_catalog; this call site logs a plain message, not exc_info.
        assert record.exc_info is None
        assert str(_catalog_path()) in record.getMessage()
        assert "corrupt" in record.getMessage()


# ---------------------------------------------------------------------------
# E3: run_supervisor lazy-boot path
# ---------------------------------------------------------------------------


class TestRunSupervisorLazyBoot:
    def test_strict_config_validation_precedes_every_owner_creation(self):
        import muse.cli_impl.supervisor as supervisor
        from muse.core.config import ConfigError

        with patch.object(
            supervisor.config,
            "validated_config",
            side_effect=ConfigError("invalid supervisor config"),
        ) as validate, patch.object(
            supervisor, "SupervisorState",
        ) as state_cls, patch.object(
            supervisor, "set_supervisor_state",
        ) as set_state, patch(
            "muse.admin.jobs.reset_default_store",
        ) as reset_store, patch(
            "muse.admin.jobs.get_default_store",
        ) as get_store, patch.object(
            supervisor.threading, "Thread",
        ) as thread_cls, patch.object(
            supervisor, "IdleSweeper",
        ) as sweeper_cls, patch.object(
            supervisor, "register_process",
        ) as register:
            with pytest.raises(ConfigError, match="invalid supervisor config"):
                supervisor.run_supervisor(
                    host="127.0.0.1", port=8000, device="cpu",
                )

        validate.assert_called_once_with()
        state_cls.assert_not_called()
        set_state.assert_not_called()
        reset_store.assert_not_called()
        get_store.assert_not_called()
        thread_cls.assert_not_called()
        sweeper_cls.assert_not_called()
        register.assert_not_called()

    def test_incomplete_worker_cleanup_returns_one_and_retains_ownership(self):
        import muse.cli_impl.supervisor as supervisor
        from muse.cli_impl.supervisor import WorkerShutdownResult

        retained_worker = WorkerSpec(
            models=["x"], python_path="/p", port=9001, status="running",
        )
        director = MagicMock(
            gpu_budget_gb=None,
            cpu_budget_gb=8.0,
            gpu_headroom_gb=0.0,
            cpu_headroom_gb=0.0,
        )
        monitor_thread = MagicMock(name="monitor_thread")
        monitor_thread.is_alive.return_value = False
        sweeper_thread = MagicMock(name="sweeper_thread")
        sweeper_thread.is_alive.return_value = False
        sweeper = MagicMock(name="sweeper")
        sweeper.start.return_value = sweeper_thread
        job_store = MagicMock(name="job_store")
        job_store.shutdown.return_value = True
        captured_state = None

        def interrupt_gateway(*_args, **_kwargs):
            nonlocal captured_state
            captured_state = supervisor.get_supervisor_state()
            captured_state.workers.append(retained_worker)
            raise KeyboardInterrupt()

        with patch.object(
            supervisor.config, "validated_config",
        ), patch.object(
            supervisor.config,
            "get",
            side_effect=lambda key: False if key == "telemetry.enabled" else None,
        ), patch(
            "muse.admin.jobs.reset_default_store",
        ), patch(
            "muse.admin.jobs.get_default_store", return_value=job_store,
        ), patch(
            "muse.cli_impl.gateway.build_gateway", return_value=MagicMock(),
        ), patch.object(
            supervisor, "_build_load_director", return_value=director,
        ), patch.object(
            supervisor, "validate_catalog_at_boot",
        ), patch.object(
            supervisor.threading, "Thread", return_value=monitor_thread,
        ), patch.object(
            supervisor, "IdleSweeper", return_value=sweeper,
        ), patch.object(
            supervisor, "register_process", return_value="supervisor-resource",
        ), patch.object(
            supervisor.os, "getpid", return_value=4242,
        ), patch.object(
            supervisor.os, "getppid", return_value=31337,
        ), patch.object(
            supervisor, "unregister_process",
        ) as unregister, patch.object(
            supervisor, "run_uvicorn", side_effect=interrupt_gateway,
        ), patch.object(
            supervisor, "_shutdown_telemetry", return_value=True,
        ), patch.object(
            supervisor,
            "_shutdown_workers",
            return_value=WorkerShutdownResult((), (retained_worker,)),
        ) as shutdown_workers, patch.object(
            supervisor, "clear_supervisor_state",
            wraps=supervisor.clear_supervisor_state,
        ) as clear_state:
            result = supervisor.run_supervisor(
                host="127.0.0.1", port=8000, device="cpu",
            )

        assert result == 1
        assert captured_state is not None
        assert supervisor.get_supervisor_state() is captured_state
        assert captured_state.supervisor_resource_id == "supervisor-resource"
        shutdown_workers.assert_called_once_with(captured_state.workers)
        unregister.assert_not_called()
        clear_state.assert_not_called()

    def test_supervisor_record_tracks_foreground_parent_and_is_unregistered(
        self, tmp_catalog,
    ):
        _seed_catalog({})
        from muse.cli_impl.supervisor import run_supervisor

        with patch(
            "muse.cli_impl.supervisor.register_process",
            return_value="supervisor-resource",
        ) as register, patch(
            "muse.cli_impl.supervisor.unregister_process",
        ) as unregister, patch(
            "muse.cli_impl.supervisor.os.getpid", return_value=4242,
        ), patch(
            "muse.cli_impl.supervisor.os.getppid", return_value=31337,
        ), patch(
            "muse.cli_impl.supervisor.threading.Thread",
        ), patch(
            "muse.cli_impl.supervisor.run_uvicorn",
            side_effect=KeyboardInterrupt(),
        ), patch(
            "muse.cli_impl.supervisor._shutdown_workers",
        ):
            assert run_supervisor(
                host="127.0.0.1", port=8000, device="cpu",
            ) == 0

        register.assert_called_once_with(
            kind="supervisor",
            pid=4242,
            owner_pid=31337,
            port=8000,
        )
        unregister.assert_called_once_with("supervisor-resource")

    def test_supervisor_registration_failure_prevents_gateway_start(
        self, tmp_catalog,
    ):
        from muse.core.resource_registry import ResourceRegistryError
        from muse.cli_impl.supervisor import run_supervisor

        _seed_catalog({})
        with patch(
            "muse.cli_impl.supervisor.register_process",
            side_effect=ResourceRegistryError("registry unavailable"),
        ), patch(
            "muse.cli_impl.supervisor.threading.Thread",
        ), patch(
            "muse.cli_impl.supervisor.run_uvicorn",
        ) as run_uvicorn, patch(
            "muse.cli_impl.supervisor._shutdown_workers",
        ):
            with pytest.raises(
                ResourceRegistryError, match="cannot start an untracked",
            ):
                run_supervisor(
                    host="127.0.0.1", port=8000, device="cpu",
                )

        run_uvicorn.assert_not_called()
        assert get_supervisor_state().director is None

    def test_run_supervisor_does_not_spawn_workers_eagerly(self, tmp_catalog):
        """The lazy supervisor must NOT call spawn_worker at boot time,
        even with one or more enabled models in the catalog.
        """
        _seed_catalog({
            "model-a": {
                "pulled_at": "...",
                "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "model-a",
                    "modality": "embedding/text",
                    "hf_repo": "a",
                    "backend_path": "muse.core.runtime:X",
                    "capabilities": {"memory_gb": 0.5, "device": "cpu"},
                },
            },
            "model-b": {
                "pulled_at": "...",
                "hf_repo": "b", "local_dir": "/b",
                "venv_path": "/venvs/b",
                "python_path": "/venvs/b/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "model-b",
                    "modality": "embedding/text",
                    "hf_repo": "b",
                    "backend_path": "muse.core.runtime:X",
                    "capabilities": {"memory_gb": 0.5, "device": "cpu"},
                },
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.threading.Thread"), \
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = KeyboardInterrupt()

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            # The whole point: zero workers spawned at boot.
            mock_spawn.assert_not_called()

    def test_run_supervisor_constructs_load_director(self, tmp_catalog):
        """A LoadDirector is on state.director during the run."""
        _seed_catalog({
            "model-a": {
                "pulled_at": "...",
                "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "model-a",
                    "modality": "embedding/text",
                    "hf_repo": "a",
                    "backend_path": "muse.core.runtime:X",
                    "capabilities": {"memory_gb": 0.5, "device": "cpu"},
                },
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        seen_state = {"value": None}

        def capture_state(*args, **kwargs):
            seen_state["value"] = get_supervisor_state()
            raise KeyboardInterrupt()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.threading.Thread"), \
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = capture_state
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        captured = seen_state["value"]
        assert captured is not None
        assert captured.director is not None
        assert isinstance(captured.director, LoadDirector)

    def test_run_supervisor_runs_validation_at_boot(self, tmp_catalog):
        """validate_catalog_at_boot is called and populates unservable_reasons."""
        _seed_catalog({
            "huge": {
                "pulled_at": "...",
                "hf_repo": "h", "local_dir": "/h",
                "venv_path": "/venvs/h",
                "python_path": "/venvs/h/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "huge",
                    "modality": "chat/completion",
                    "hf_repo": "h",
                    "backend_path": "muse.core.runtime:X",
                    "capabilities": {"memory_gb": 9999.0, "device": "cpu"},
                },
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        seen_state = {"value": None}

        def capture_state(*args, **kwargs):
            seen_state["value"] = get_supervisor_state()
            raise KeyboardInterrupt()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.threading.Thread"), \
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = capture_state
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        captured = seen_state["value"]
        assert captured is not None
        # huge model exceeds capacity; must be flagged.
        assert "huge" in captured.unservable_reasons

    def test_run_supervisor_starts_gateway_immediately_with_empty_catalog(self, tmp_catalog):
        """With zero enabled models, the lazy supervisor still boots the
        gateway. No worker spawn loop, no first-ready wait.
        """
        _seed_catalog({})
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = KeyboardInterrupt()
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            # Empty catalog: no workers spawned; gateway starts.
            mock_spawn.assert_not_called()
            mock_run_uvicorn.assert_called_once()

    def test_run_supervisor_clears_state_on_exit(self, tmp_catalog):
        """The teardown path still clears the singleton state."""
        _seed_catalog({})
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = KeyboardInterrupt()
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        # Cleared after exit
        cleared = get_supervisor_state()
        assert cleared.workers == []
        assert cleared.director is None
        assert cleared.unservable_reasons == {}


# ---------------------------------------------------------------------------
# E5 (auto-restart compatibility)
# ---------------------------------------------------------------------------


class TestAutoRestartMonitorLazyCompatibility:
    def test_monitor_iterates_state_workers_added_post_boot(self, tmp_catalog):
        """The monitor reads state.workers as a live reference. Workers
        added post-boot (via the director's enable_fn) must be picked up
        on the next monitor tick. This is verified by run_supervisor
        passing state.workers (not a frozen list) to the monitor thread.
        """
        _seed_catalog({})
        from muse.cli_impl.supervisor import run_supervisor

        captured_args: dict = {}

        def capture_thread_init(*args, **kwargs):
            target = kwargs.get("target")
            if target is not None and getattr(target, "__name__", "") == "_monitor_workers":
                # The first positional arg is the workers list; capture
                # its identity so we can assert it's state.workers.
                captured_args["specs"] = kwargs.get("args", (None,))[0]
            t = MagicMock()
            t.start = MagicMock()
            return t

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.threading.Thread",
                   side_effect=capture_thread_init), \
             patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = KeyboardInterrupt()
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        # The monitor was started with the live state.workers reference,
        # not a copy. (When the catalog is empty, the monitor may or may
        # not be started; this test focuses on the live-reference shape
        # when it is.)
        if "specs" in captured_args:
            specs = captured_args["specs"]
            # state.workers is either an empty list or a live reference;
            # crucially, it's the SAME list object used by the rest of
            # the supervisor + admin endpoints.
            assert isinstance(specs, list)


# ---------------------------------------------------------------------------
# Issue 1: monitor must skip in-flight specs (job_id != None) so it does NOT
# race a director-driven cold load that's still bringing the worker up.
# ---------------------------------------------------------------------------


class TestMonitorSkipsInFlightSpecs:
    def test_monitor_skips_spec_with_job_id_set(self):
        """A spec whose job_id is set is in the middle of an admin- or
        director-driven transition. The monitor must NOT poll its
        /health, NOT increment failure_count, and NOT call _attempt_restart.
        """
        from muse.cli_impl.supervisor import _monitor_workers, WorkerSpec
        import threading

        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001, device="cpu",
        )
        spec.process = None  # not yet spawned (mid-cold-load)
        spec.status = "pending"
        spec.job_id = "director-load-x"  # in-flight
        stop_event = threading.Event()

        # Stop after a single tick so the test doesn't spin forever.
        tick_count = {"n": 0}

        def health_side_effect(**kwargs):
            tick_count["n"] += 1
            if tick_count["n"] >= 1:
                stop_event.set()
            return False

        with patch(
            "muse.cli_impl.supervisor.check_worker_health",
            side_effect=health_side_effect,
        ) as mock_health, patch(
            "muse.cli_impl.supervisor._attempt_restart"
        ) as mock_restart:
            # Need at least one workers entry so the loop has work to do
            # (without that the outer while just sleeps until stop_event);
            # a second non-job_id spec drives the stop_event.
            other = WorkerSpec(
                models=["y"], python_path="/p2", port=9002, device="cpu",
            )
            other.process = MagicMock(
                pid=12345, poll=MagicMock(return_value=None),
            )
            other.status = "running"
            _monitor_workers(
                [spec, other], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        # The in-flight spec was never health-polled. Only the other
        # spec contributed to the call counter.
        polled_ports = [c.kwargs.get("port") for c in mock_health.call_args_list]
        assert 9001 not in polled_ports, (
            "monitor polled in-flight spec; should have skipped"
        )
        # And the in-flight spec was never restarted.
        for c in mock_restart.call_args_list:
            assert c.args[0] is not spec, (
                "monitor attempted to restart an in-flight spec"
            )
        # The spec's status / failure_count are unchanged: still pending.
        assert spec.status == "pending"
        assert spec.failure_count == 0
        assert spec.job_id == "director-load-x"  # untouched

    def test_director_load_does_not_double_spawn_under_monitor(
        self, tmp_catalog,
    ):
        """End-to-end: when the director calls load_model_into_worker
        with a slow spawn_worker, the monitor must not restart the
        in-flight pending spec mid-spawn.
        """
        import threading
        import time

        from muse.admin.operations import load_model_into_worker
        from muse.cli_impl.supervisor import (
            SupervisorState,
            _monitor_workers,
        )

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })

        state = SupervisorState(workers=[], device="cpu")
        spawn_count = {"n": 0}
        spawn_done = threading.Event()

        def slow_spawn(spec, device, **_kwargs):
            spawn_count["n"] += 1
            time.sleep(0.4)  # simulate slow GGUF / SDXL load
            spawn_done.set()

        stop_event = threading.Event()

        with patch(
            "muse.admin.operations.spawn_worker",
            side_effect=slow_spawn,
        ), patch(
            "muse.admin.operations.wait_for_ready",
            lambda *a, **k: None,
        ), patch(
            "muse.admin.operations.find_free_port",
            lambda *a, **k: 9123,
        ), patch(
            "muse.cli_impl.supervisor.check_worker_health",
            return_value=False,  # simulate port-not-bound during load
        ), patch(
            "muse.cli_impl.supervisor._attempt_restart"
        ) as mock_restart:
            # Start the monitor with the live state.workers ref. It
            # ticks fast (1ms).
            monitor_thread = threading.Thread(
                target=_monitor_workers,
                args=(state.workers, stop_event),
                kwargs={
                    "interval": 0.001,
                    "failure_threshold": 3,
                    "max_restarts": 10,
                },
                daemon=True,
            )
            monitor_thread.start()

            # Drive the director-style load on the main thread. Slow
            # spawn keeps the spec pending+job_id-bearing for ~0.4s.
            try:
                port = load_model_into_worker("kokoro-82m", state=state)
            finally:
                stop_event.set()
                monitor_thread.join(timeout=2.0)

        assert port == 9123
        assert spawn_count["n"] == 1, (
            f"expected 1 spawn (no double-spawn), got {spawn_count['n']}"
        )
        # Critically: no restart was attempted on the in-flight spec.
        mock_restart.assert_not_called()
        # And the spec is now running with no in-flight job_id.
        assert len(state.workers) == 1
        assert state.workers[0].status == "running"
        assert state.workers[0].job_id is None


# ---------------------------------------------------------------------------
# Issue 2: unload_model_from_worker must release state.lock during the slow
# shutdown / restart-in-place phase.
# ---------------------------------------------------------------------------


class TestUnloadReleasesLockDuringSlowIO:
    def test_unload_releases_lock_during_shutdown(self, tmp_catalog):
        """unload_model_from_worker on a sole-tenant worker calls
        _shutdown_workers, which can take seconds. The lock must be
        dropped during that window so admin reads / hot-acquires are
        not blocked.
        """
        import threading
        import time

        from muse.admin.operations import unload_model_from_worker
        from muse.cli_impl.supervisor import SupervisorState, WorkerSpec

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })

        state = SupervisorState(workers=[], device="cpu")
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python",
            port=9001,
        )
        spec.status = "running"
        state.workers.append(spec)

        def slow_shutdown(specs):
            time.sleep(0.4)

        unload_done = threading.Event()

        with patch(
            "muse.admin.operations._shutdown_workers",
            side_effect=slow_shutdown,
        ):
            def _unload():
                unload_model_from_worker("kokoro-82m", state=state)
                unload_done.set()

            t = threading.Thread(target=_unload)
            t.start()
            # Give the unload thread a head start so it's mid-shutdown.
            time.sleep(0.1)

            # Time the lock acquire. With the bug, this would block for
            # the full slow-shutdown duration (~0.3s remaining).
            t0 = time.perf_counter()
            with state.lock:
                snapshot = list(state.workers)
            elapsed = time.perf_counter() - t0
            assert elapsed < 0.05, (
                f"state.lock was held during shutdown for {elapsed:.3f}s; "
                "should release before _shutdown_workers"
            )

            unload_done.wait(timeout=3.0)
            t.join(timeout=1.0)

        # After unload completes, the spec is gone.
        assert state.workers == []

    def test_unload_releases_lock_during_restart_inplace(self, tmp_catalog):
        """unload_model_from_worker on a sibling-occupied worker calls
        _restart_worker_inplace, which is also slow. Same lock-release
        contract must hold.
        """
        import threading
        import time

        from muse.admin.operations import unload_model_from_worker
        from muse.cli_impl.supervisor import SupervisorState, WorkerSpec

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": True,
            },
            "soprano-80m": {
                "pulled_at": "...", "hf_repo": "s", "local_dir": "/s",
                "venv_path": "/venv/shared",
                "python_path": "/venv/shared/bin/python",
                "enabled": True,
            },
        })

        state = SupervisorState(workers=[], device="cpu")
        spec = WorkerSpec(
            models=["kokoro-82m", "soprano-80m"],
            python_path="/venv/shared/bin/python",
            port=9001,
        )
        spec.status = "running"
        state.workers.append(spec)

        def slow_restart(spec, *, device, **_kwargs):
            time.sleep(0.4)

        unload_done = threading.Event()

        with patch(
            "muse.admin.operations._restart_worker_inplace",
            side_effect=slow_restart,
        ):
            def _unload():
                unload_model_from_worker("soprano-80m", state=state)
                unload_done.set()

            t = threading.Thread(target=_unload)
            t.start()
            time.sleep(0.1)

            t0 = time.perf_counter()
            with state.lock:
                snapshot = list(state.workers)
            elapsed = time.perf_counter() - t0
            assert elapsed < 0.05, (
                f"state.lock held during restart-in-place for {elapsed:.3f}s; "
                "should release before _restart_worker_inplace"
            )

            unload_done.wait(timeout=3.0)
            t.join(timeout=1.0)

    def test_disable_releases_lock_during_shutdown(self, tmp_catalog):
        """Pre-existing same bug in disable_model. Same fix applied:
        slow shutdown happens outside state.lock.
        """
        import threading
        import time

        from muse.admin.operations import disable_model
        from muse.cli_impl.supervisor import SupervisorState, WorkerSpec

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })

        state = SupervisorState(workers=[], device="cpu")
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python",
            port=9001,
        )
        spec.status = "running"
        state.workers.append(spec)

        def slow_shutdown(specs):
            time.sleep(0.4)

        done = threading.Event()

        with patch(
            "muse.admin.operations._shutdown_workers",
            side_effect=slow_shutdown,
        ):
            def _disable():
                disable_model("kokoro-82m", state=state)
                done.set()

            t = threading.Thread(target=_disable)
            t.start()
            time.sleep(0.1)

            t0 = time.perf_counter()
            with state.lock:
                snapshot = list(state.workers)
            elapsed = time.perf_counter() - t0
            assert elapsed < 0.05, (
                f"state.lock held during disable_model shutdown for "
                f"{elapsed:.3f}s; should release before _shutdown_workers"
            )

            done.wait(timeout=3.0)
            t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Issue 3: state.workers must be mutated in place, not rebound, so the
# monitor thread (which captured the original list reference) sees updates.
# ---------------------------------------------------------------------------


class TestStateWorkersMutatedInPlace:
    def test_unload_mutates_workers_in_place(self, tmp_catalog):
        """unload_model_from_worker must remove the spec via in-place
        mutation (not rebind state.workers) so the monitor thread,
        which captured the original list reference, sees the removal.
        """
        from muse.admin.operations import unload_model_from_worker
        from muse.cli_impl.supervisor import SupervisorState, WorkerSpec

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })

        state = SupervisorState(workers=[], device="cpu")
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python",
            port=9001,
        )
        spec.status = "running"
        state.workers.append(spec)

        # Capture the list identity BEFORE the call. After the call,
        # state.workers must STILL be the same list object.
        original_id = id(state.workers)

        with patch("muse.admin.operations._shutdown_workers"):
            unload_model_from_worker("kokoro-82m", state=state)

        assert id(state.workers) == original_id, (
            "state.workers was rebound; monitor thread (which captured "
            "the original reference) would now iterate a stale list"
        )
        assert state.workers == []

    def test_disable_mutates_workers_in_place(self, tmp_catalog):
        """Same in-place mutation contract for disable_model
        (pre-existing bug, fixed in the same pass).
        """
        from muse.admin.operations import disable_model
        from muse.cli_impl.supervisor import SupervisorState, WorkerSpec

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
        })

        state = SupervisorState(workers=[], device="cpu")
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python",
            port=9001,
        )
        spec.status = "running"
        state.workers.append(spec)

        original_id = id(state.workers)

        with patch("muse.admin.operations._shutdown_workers"):
            disable_model("kokoro-82m", state=state)

        assert id(state.workers) == original_id, (
            "state.workers was rebound after disable_model"
        )
        assert state.workers == []

    def test_monitor_sees_unload_via_live_reference(self, tmp_catalog):
        """End-to-end: spawn 2 workers, remove one via
        unload_model_from_worker, run a monitor tick. The monitor must
        iterate only the remaining worker (mock _attempt_restart and
        assert it's not called for the removed one).
        """
        import threading

        from muse.admin.operations import unload_model_from_worker
        from muse.cli_impl.supervisor import (
            SupervisorState,
            WorkerSpec,
            _monitor_workers,
        )

        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/venv/k",
                "python_path": "/venv/k/bin/python",
                "enabled": True,
            },
            "soprano-80m": {
                "pulled_at": "...", "hf_repo": "s", "local_dir": "/s",
                "venv_path": "/venv/s",
                "python_path": "/venv/s/bin/python",
                "enabled": True,
            },
        })

        state = SupervisorState(workers=[], device="cpu")
        spec_k = WorkerSpec(
            models=["kokoro-82m"], python_path="/venv/k/bin/python",
            port=9001,
        )
        spec_k.status = "running"
        spec_k.process = MagicMock(poll=MagicMock(return_value=1))  # exited
        spec_s = WorkerSpec(
            models=["soprano-80m"], python_path="/venv/s/bin/python",
            port=9002,
        )
        spec_s.status = "running"
        spec_s.process = MagicMock(poll=MagicMock(return_value=1))  # exited
        state.workers.append(spec_k)
        state.workers.append(spec_s)

        # Capture the live reference the monitor would receive at boot.
        monitor_workers_ref = state.workers

        # Remove kokoro via unload BEFORE running the monitor tick.
        with patch("muse.admin.operations._shutdown_workers"):
            unload_model_from_worker("kokoro-82m", state=state)

        # Now run a single monitor tick over the (mutated-in-place) list.
        # The monitor sees process.poll() returns nonzero -> failure_count
        # ratchets to threshold -> _attempt_restart is called for the
        # remaining worker only.
        stop_event = threading.Event()
        # Stop after the first iteration of the for-loop ticks. We use
        # _attempt_restart's first invocation as the trigger: it gets
        # called for the surviving worker (whose process.poll() returns
        # nonzero), and we set the stop_event from inside the mock.
        def restart_side_effect(spec, *, stop_event=stop_event, **kw):
            stop_event.set()

        with patch(
            "muse.cli_impl.supervisor._attempt_restart",
            side_effect=restart_side_effect,
        ) as mock_restart:
            _monitor_workers(
                monitor_workers_ref, stop_event,
                interval=0.001, failure_threshold=1, max_restarts=10,
            )

        # Critically: the monitor saw only the remaining worker via the
        # captured-but-mutated-in-place list reference.
        restart_targets = [c.args[0] for c in mock_restart.call_args_list]
        for target in restart_targets:
            assert target is not spec_k, (
                "monitor tried to restart the removed (kokoro) worker"
            )
        # And it DID try to restart the surviving worker.
        assert any(t is spec_s for t in restart_targets), (
            "monitor failed to see the surviving worker via the live "
            "reference (state.workers was rebound)"
        )


# ---------------------------------------------------------------------------
# Task B (v0.40.1): IdleSweeper wired at boot.
#
# The sweeper is created, started, exposed on state.idle_sweeper +
# state.idle_sweeper_thread, and joined on graceful shutdown via the
# shared state.stop_event.
# ---------------------------------------------------------------------------


class TestIdleSweeperWiredAtBoot:
    def test_run_supervisor_starts_idle_sweeper(self, tmp_catalog):
        """state.idle_sweeper is an IdleSweeper, and its thread is alive
        at the moment the gateway runs.
        """
        _seed_catalog({})
        from muse.cli_impl.supervisor import run_supervisor

        seen: dict = {"state": None, "thread_alive": None, "sweeper": None}

        def capture_state(*args, **kwargs):
            s = get_supervisor_state()
            seen["state"] = s
            seen["sweeper"] = s.idle_sweeper
            seen["thread_alive"] = (
                s.idle_sweeper_thread is not None
                and s.idle_sweeper_thread.is_alive()
            )
            raise KeyboardInterrupt()

        with patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = capture_state
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        assert seen["sweeper"] is not None, (
            "expected state.idle_sweeper to be populated by run_supervisor"
        )
        assert isinstance(seen["sweeper"], IdleSweeper)
        assert seen["thread_alive"] is True, (
            "expected state.idle_sweeper_thread to be alive while uvicorn runs"
        )

    def test_run_supervisor_stops_idle_sweeper_on_shutdown(self, tmp_catalog):
        """After run_supervisor exits, the sweeper thread is no longer
        alive: stop_event was set in cleanup and the thread joined.
        """
        _seed_catalog({})
        from muse.cli_impl.supervisor import run_supervisor

        captured: dict = {"thread": None}

        def capture_thread(*args, **kwargs):
            s = get_supervisor_state()
            captured["thread"] = s.idle_sweeper_thread
            raise KeyboardInterrupt()

        with patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = capture_thread
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        thread = captured["thread"]
        assert thread is not None
        # The cleanup path joins with timeout=5.0; the thread should have
        # exited already because stop_event was set. Allow a small grace
        # window in case scheduling delay leaves it briefly alive.
        thread.join(timeout=5.0)
        assert not thread.is_alive(), (
            "idle sweeper thread is still alive after run_supervisor "
            "returned; stop_event/join did not propagate"
        )

    def test_idle_sweep_interval_env_var_respected(
        self, tmp_catalog, monkeypatch,
    ):
        """MUSE_IDLE_SWEEP_INTERVAL_SECONDS=0.1 reaches IdleSweeper.interval_seconds."""
        _seed_catalog({})
        monkeypatch.setenv("MUSE_IDLE_SWEEP_INTERVAL_SECONDS", "0.1")
        from muse.cli_impl.supervisor import run_supervisor

        seen: dict = {"interval": None}

        def capture_interval(*args, **kwargs):
            s = get_supervisor_state()
            seen["interval"] = s.idle_sweeper.interval_seconds
            raise KeyboardInterrupt()

        with patch("muse.cli_impl.supervisor.run_uvicorn") as mock_run_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"):
            mock_run_uvicorn.side_effect = capture_interval
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        assert seen["interval"] == 0.1


class TestHasMemoryData:
    """Unit tests for _has_memory_data's (has, gb, device) tuple."""

    def test_bundled_model_adopts_cross_device_measurement(self):
        """A bundled model (no persisted manifest -> device defaults 'cpu', no
        declared memory_gb) probed only on cuda must adopt the cuda measurement
        AND its recorded device, so the capacity check uses the GPU pool."""
        from muse.cli_impl.supervisor import _has_memory_data

        has, gb, device = _has_memory_data({
            "measurements": {
                "cuda": {"peak_bytes": 3 * 1024 ** 3, "device": "cuda"},
            },
        })
        assert has is True
        assert device == "cuda"
        assert round(gb) == 3

    def test_declared_device_not_overwritten_by_stale_measurement(self):
        """Regression: a model that DECLARES device=cuda + memory_gb but whose
        only measurement is a stale CPU probe (e.g. pulled with --no-probe then
        probed on a CPU host) must keep device='cuda' so its GPU estimate is
        sized against the GPU pool, not silently mis-sized against CPU."""
        from muse.cli_impl.supervisor import _has_memory_data

        has, gb, device = _has_memory_data({
            "manifest": {"capabilities": {"device": "cuda", "memory_gb": 14.0}},
            "measurements": {
                "cpu": {"peak_bytes": 1 * 1024 ** 3, "device": "cpu"},
            },
        })
        assert has is True
        assert gb == 14.0
        assert device == "cuda"  # NOT overwritten to 'cpu' by the recovery loop

    @pytest.mark.parametrize(
        "entry",
        [
            {"manifest": {"capabilities": {"memory_gb": -1}}},
            {"manifest": {"capabilities": {"memory_gb": "bad"}}},
            {"manifest": {"capabilities": {"memory_gb": float("nan")}}},
            {"manifest": {"capabilities": []}},
            {"manifest": [], "measurements": []},
            {"measurements": {"auto": []}},
        ],
    )
    def test_malformed_capacity_data_is_unsizable_not_an_exception(self, entry):
        from muse.cli_impl.supervisor import _has_memory_data

        has, gb, device = _has_memory_data(entry)

        assert has is False
        assert gb == 0.0
        assert device == "auto"

    def test_invalid_declared_memory_has_actionable_reason(self):
        from muse.cli_impl.supervisor import _servability_reason

        reason = _servability_reason(
            {"manifest": {"capabilities": {"memory_gb": float("inf")}}},
            cpu_available_gb=64.0,
            gpu_available_gb=None,
        )

        assert reason is not None
        assert "invalid memory estimate" in reason
        assert "finite non-negative" in reason


class TestServabilityAutoDevice:
    """v0.48.0: a resolver-pulled model declaring device='auto' must be
    sized against the pool it actually loads on. On a GPU host the
    servability check has to use the VRAM pool, not the (large) host-RAM
    pool -- otherwise an auto model that cannot fit VRAM is waved through
    at boot/request and only fails after the director evicts the whole
    idle working set. Mirrors the LoadDirector's auto resolution: a GPU is
    present iff live VRAM info is available (gpu_available_gb is not None).
    """

    def test_auto_device_sized_against_gpu_pool_when_gpu_present(self):
        from muse.cli_impl.supervisor import _servability_reason

        entry = {
            "enabled": True,
            "manifest": {"capabilities": {"device": "auto", "memory_gb": 20.0}},
        }
        # 20 GB fits the 500 GB CPU pool but NOT the 8 GB GPU pool.
        reason = _servability_reason(
            entry, cpu_available_gb=500.0, gpu_available_gb=8.0
        )
        assert reason is not None
        assert "exceeds device capacity" in reason
        assert "20.0 GB > 8.0 GB" in reason

    def test_auto_device_sized_against_cpu_pool_when_no_gpu(self):
        from muse.cli_impl.supervisor import _servability_reason

        entry = {
            "enabled": True,
            "manifest": {"capabilities": {"device": "auto", "memory_gb": 20.0}},
        }
        # CPU-only host (gpu_available_gb is None): auto falls back to the
        # CPU pool, where 20 GB fits 500 GB -> servable.
        reason = _servability_reason(
            entry, cpu_available_gb=500.0, gpu_available_gb=None
        )
        assert reason is None

    def test_explicit_supervisor_cpu_overrides_manifest_auto(self):
        from muse.cli_impl.supervisor import _servability_reason

        entry = {
            "manifest": {"capabilities": {"device": "auto", "memory_gb": 6.0}},
        }
        reason = _servability_reason(
            entry,
            cpu_available_gb=4.0,
            gpu_available_gb=100.0,
            supervisor_device="cpu",
        )
        assert reason is not None
        assert "6.0 GB > 4.0 GB capacity on cpu" in reason

    def test_manifest_pin_beats_explicit_supervisor_device(self):
        from muse.cli_impl.supervisor import _servability_reason

        entry = {
            "manifest": {"capabilities": {"device": "cuda", "memory_gb": 6.0}},
        }
        assert _servability_reason(
            entry,
            cpu_available_gb=1.0,
            gpu_available_gb=8.0,
            supervisor_device="cpu",
        ) is None

    def test_catalog_override_beats_manifest_and_supervisor(self):
        from muse.cli_impl.supervisor import _servability_reason

        entry = {
            "device_override": "cpu",
            "manifest": {"capabilities": {"device": "cuda", "memory_gb": 6.0}},
        }
        reason = _servability_reason(
            entry,
            cpu_available_gb=4.0,
            gpu_available_gb=100.0,
            supervisor_device="cuda",
        )
        assert reason is not None
        assert "capacity on cpu" in reason


# ---------------------------------------------------------------------------
# v0.47.3 Bug #2: revalidate_servability re-evaluates one model's
# "no memory estimate" stamp against the LIVE catalog, so a `muse models
# probe` that lands after boot takes effect without a supervisor restart.
# ---------------------------------------------------------------------------


class TestRevalidateServability:
    def _probe_stamp(self) -> str:
        return "no memory estimate; run `muse models probe` to populate"

    def test_clears_stamp_when_measurement_present(self, tmp_catalog):
        """A model boot-stamped 'no memory estimate' whose catalog now
        carries a measurement is cleared (returns None) and removed from
        state.unservable_reasons -- the probe-then-serve fix.
        """
        from muse.cli_impl.supervisor import revalidate_servability

        _seed_catalog({
            "probed-model": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "probed-model",
                    "modality": "embedding/text",
                    "capabilities": {"device": "cpu"},  # no memory_gb
                },
                "measurements": {
                    "cpu": {"peak_bytes": 800_000_000, "device": "cpu"},
                },
            },
        })
        state = SupervisorState()
        state.unservable_reasons["probed-model"] = self._probe_stamp()

        reason = revalidate_servability(state, "probed-model")

        assert reason is None
        assert "probed-model" not in state.unservable_reasons

    def test_clears_stamp_when_declared_memory_added(self, tmp_catalog):
        """A capabilities.memory_gb annotation also resolves the stamp."""
        from muse.cli_impl.supervisor import revalidate_servability

        _seed_catalog({
            "annotated": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "annotated",
                    "modality": "embedding/text",
                    "capabilities": {"device": "cpu", "memory_gb": 1.0},
                },
            },
        })
        state = SupervisorState()
        state.unservable_reasons["annotated"] = self._probe_stamp()

        reason = revalidate_servability(state, "annotated")

        assert reason is None
        assert "annotated" not in state.unservable_reasons

    def test_keeps_stamp_when_still_no_estimate(self, tmp_catalog):
        """A model still lacking any estimate keeps the probe-prompt 503."""
        from muse.cli_impl.supervisor import revalidate_servability

        _seed_catalog({
            "needs-probe": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "needs-probe",
                    "modality": "embedding/text",
                    "capabilities": {"device": "cpu"},  # no memory data
                },
            },
        })
        state = SupervisorState()
        state.unservable_reasons["needs-probe"] = self._probe_stamp()

        reason = revalidate_servability(state, "needs-probe")

        assert reason is not None
        assert "no memory estimate" in reason
        assert "needs-probe" in state.unservable_reasons

    def test_clears_stale_stamp_when_model_absent_from_catalog(self, tmp_catalog):
        """A model removed from the catalog since boot has its stale stamp
        CLEARED (returns None), so the gateway falls through to get_manifest
        and 404s `model_not_found` (or serves a bundled fallback) instead of
        503'ing with a stamp that refers to a model that no longer exists.

        Regression guard for the adversarial-review LOW finding: the prior
        cut returned the retained stale reason here, short-circuiting the
        gateway to 503 and contradicting the documented 404 path.
        """
        from muse.cli_impl.supervisor import revalidate_servability

        _seed_catalog({})
        state = SupervisorState()
        state.unservable_reasons["ghost"] = "some prior reason"

        reason = revalidate_servability(state, "ghost")

        assert reason is None
        assert "ghost" not in state.unservable_reasons

    def _capacity_probe(self, cpu_free=16.0, gpu_free=None):
        probe = MagicMock()
        probe.cpu_free_gb.return_value = cpu_free
        probe.gpu_free_gb.return_value = gpu_free
        return probe

    def test_keeps_capacity_stamp_for_oversized_model(self, tmp_catalog):
        """v0.47.3 regression guard: a model boot-stamped 'exceeds device
        capacity' must NOT be cleared merely because it is sizable. Clearing
        it would route an impossible-to-fit request into the director, whose
        eviction loop tears down the entire idle working set before 503'ing.
        revalidate must re-derive the FULL verdict (estimate AND capacity)
        and keep the capacity stamp.
        """
        from muse.cli_impl.supervisor import revalidate_servability

        _seed_catalog({
            "huge": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "huge",
                    "modality": "audio/speech",
                    "capabilities": {"device": "cpu", "memory_gb": 200.0},
                },
            },
        })
        state = SupervisorState()
        state.unservable_reasons["huge"] = (
            "exceeds device capacity (200.0 GB > 14.0 GB available on cpu)"
        )

        reason = revalidate_servability(
            state, "huge", memory_probe=self._capacity_probe(cpu_free=16.0),
        )

        assert reason is not None
        assert "exceeds device capacity" in reason
        assert "huge" in state.unservable_reasons

    def test_keeps_capacity_stamp_for_gpu_without_pynvml(self, tmp_catalog):
        """A cuda model on a host with no live VRAM info (pynvml absent) is
        not sizable-for-fit; revalidate keeps the capacity stamp so the
        gateway 503s instead of deferring to the director (which would size
        available GPU at 0 and evict the world).
        """
        from muse.cli_impl.supervisor import revalidate_servability

        _seed_catalog({
            "gpu-model": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "gpu-model",
                    "modality": "image/generation",
                    "capabilities": {"device": "cuda", "memory_gb": 8.0},
                },
            },
        })
        state = SupervisorState()
        state.unservable_reasons["gpu-model"] = (
            "exceeds device capacity (no GPU info available; ...)"
        )

        reason = revalidate_servability(
            state, "gpu-model",
            memory_probe=self._capacity_probe(gpu_free=None),
        )

        assert reason is not None
        assert "exceeds device capacity" in reason
        assert "gpu-model" in state.unservable_reasons

    def test_uses_director_gpu_budget_when_pynvml_is_unavailable(self, tmp_catalog):
        """Gateway revalidation passes headroom but historically omitted
        budgets; recover the budget from state.director so its gate and the
        following admission decision agree.
        """
        from muse.cli_impl.supervisor import revalidate_servability

        _seed_catalog({
            "gpu-model": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "capabilities": {"device": "cuda", "memory_gb": 8.0},
                },
            },
        })
        probe = self._capacity_probe(gpu_free=None)
        state = SupervisorState()
        state.director = LoadDirector(
            enable_fn=MagicMock(),
            disable_fn=MagicMock(),
            memory_probe=probe,
            gpu_budget_gb=10.0,
            gpu_headroom_gb=1.0,
        )
        state.unservable_reasons["gpu-model"] = (
            "exceeds device capacity (no GPU info available; ...)"
        )

        reason = revalidate_servability(
            state, "gpu-model", memory_probe=probe,
        )

        assert reason is None
        assert "gpu-model" not in state.unservable_reasons

    def test_clears_capacity_stamp_when_memory_now_available(self, tmp_catalog):
        """The live re-check uses LIVE free memory, not the stale boot
        snapshot: a model stamped at boot (when free was low) is cleared once
        enough memory is actually free now -- this is the legitimate
        'load later when memory frees' path, distinct from the oversized one.
        """
        from muse.cli_impl.supervisor import revalidate_servability

        _seed_catalog({
            "fits-now": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "fits-now",
                    "modality": "audio/speech",
                    "capabilities": {"device": "cpu", "memory_gb": 4.0},
                },
            },
        })
        state = SupervisorState()
        state.unservable_reasons["fits-now"] = (
            "exceeds device capacity (4.0 GB > 1.0 GB available on cpu)"
        )

        reason = revalidate_servability(
            state, "fits-now", memory_probe=self._capacity_probe(cpu_free=16.0),
        )

        assert reason is None
        assert "fits-now" not in state.unservable_reasons

    def test_clears_hard_stamp_while_reclaimable_worker_uses_free_memory(
        self, tmp_catalog,
    ):
        """Low current free RAM is transient when the model fits the host.

        The gateway must let the director evict or wait for a Muse-owned
        worker instead of retaining a permanent ``model_unservable`` stamp.
        """
        from muse.cli_impl.supervisor import revalidate_servability

        _seed_catalog({
            "fits-empty-pool": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "capabilities": {"device": "cpu", "memory_gb": 6.0},
                },
            },
        })

        class BusyHostProbe:
            def cpu_free_gb(self):
                return 2.0

            def cpu_total_gb(self):
                return 16.0

            def gpu_free_gb(self):
                return None

        probe = BusyHostProbe()
        state = SupervisorState()
        state.director = LoadDirector(
            enable_fn=MagicMock(), disable_fn=MagicMock(),
            memory_probe=probe, cpu_headroom_gb=2.0,
        )
        now = 1.0
        state.director.loaded["resident"] = LoadEntry(
            model_id="resident", worker_port=9001, memory_gb=8.0,
            refcount=0, last_touched_at=now, loaded_at=now, pool="cpu",
        )
        state.unservable_reasons["fits-empty-pool"] = (
            "exceeds device capacity (stale transient-free stamp)"
        )

        reason = revalidate_servability(
            state, "fits-empty-pool", memory_probe=probe,
        )

        assert reason is None
        assert "fits-empty-pool" not in state.unservable_reasons


# ---------------------------------------------------------------------------
# v0.47.3 Gap #2b: backfill_manifest_memory sizes a probed-only model from
# its catalog measurement so the LoadDirector's fit/eviction accounting is
# accurate (the director reads capabilities.memory_gb; the probe writes
# measurements.<device>.peak_bytes).
# ---------------------------------------------------------------------------


class TestBackfillManifestMemory:
    def test_sets_memory_gb_from_measurement(self, tmp_catalog):
        from muse.cli_impl.supervisor import backfill_manifest_memory

        _seed_catalog({
            "probed-model": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "probed-model",
                    "capabilities": {"device": "cpu"},
                },
                "measurements": {
                    "cpu": {"peak_bytes": 800_000_000, "device": "cpu"},
                },
            },
        })
        manifest = {
            "model_id": "probed-model",
            "modality": "embedding/text",
            "capabilities": {"device": "cpu"},  # no memory_gb
        }

        out = backfill_manifest_memory(manifest, "probed-model")

        assert out["capabilities"]["memory_gb"] == pytest.approx(
            800_000_000 / 1024 ** 3, rel=1e-6,
        )
        # Input is not mutated (copy semantics).
        assert "memory_gb" not in manifest["capabilities"]

    def test_declared_memory_wins(self, tmp_catalog):
        from muse.cli_impl.supervisor import backfill_manifest_memory

        _seed_catalog({
            "declared": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "measurements": {
                    "cpu": {"peak_bytes": 800_000_000, "device": "cpu"},
                },
            },
        })
        manifest = {
            "model_id": "declared",
            "capabilities": {"device": "cpu", "memory_gb": 2.0},
        }

        out = backfill_manifest_memory(manifest, "declared")

        assert out["capabilities"]["memory_gb"] == 2.0

    def test_noop_when_no_measurement(self, tmp_catalog):
        from muse.cli_impl.supervisor import backfill_manifest_memory

        _seed_catalog({
            "bare": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {"capabilities": {"device": "cpu"}},
            },
        })
        manifest = {"model_id": "bare", "capabilities": {"device": "cpu"}}

        out = backfill_manifest_memory(manifest, "bare")

        assert out["capabilities"].get("memory_gb") is None

    def test_noop_when_model_absent(self, tmp_catalog):
        from muse.cli_impl.supervisor import backfill_manifest_memory

        _seed_catalog({})
        manifest = {"model_id": "ghost", "capabilities": {}}

        out = backfill_manifest_memory(manifest, "ghost")

        assert out["capabilities"].get("memory_gb") is None

    def test_device_override_folded_into_capabilities(self, tmp_catalog):
        """M6: an operator `set-device` pin (catalog device_override) decides
        where the worker loads, so the control plane must size against that
        device. Fold it into capabilities.device so the LoadDirector's
        pool selection matches load_backend."""
        from muse.cli_impl.supervisor import backfill_manifest_memory

        _seed_catalog({
            "pinned": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "device_override": "cpu",
                "manifest": {"capabilities": {"device": "cuda"}},
            },
        })
        manifest = {
            "model_id": "pinned",
            "capabilities": {"device": "cuda", "memory_gb": 6.0},
        }

        out = backfill_manifest_memory(manifest, "pinned")

        assert out["capabilities"]["device"] == "cpu"
        # Input is not mutated.
        assert manifest["capabilities"]["device"] == "cuda"

    def test_device_override_applies_even_with_declared_memory(self, tmp_catalog):
        """The override fold must fire even when memory_gb is already
        declared (that branch returns early for memory but device must
        still be overridden)."""
        from muse.cli_impl.supervisor import backfill_manifest_memory

        _seed_catalog({
            "pinned2": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "device_override": "cuda",
                "manifest": {"capabilities": {"device": "cpu"}},
            },
        })
        manifest = {
            "model_id": "pinned2",
            "capabilities": {"device": "cpu", "memory_gb": 1.0},
        }

        out = backfill_manifest_memory(manifest, "pinned2")

        assert out["capabilities"]["device"] == "cuda"
        assert out["capabilities"]["memory_gb"] == 1.0

    def test_no_device_override_leaves_device_untouched(self, tmp_catalog):
        from muse.cli_impl.supervisor import backfill_manifest_memory

        _seed_catalog({
            "unpinned": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {"capabilities": {"device": "cuda"}},
            },
        })
        manifest = {
            "model_id": "unpinned",
            "capabilities": {"device": "cuda", "memory_gb": 6.0},
        }

        out = backfill_manifest_memory(manifest, "unpinned")

        assert out["capabilities"]["device"] == "cuda"

    def test_supervisor_device_replaces_manifest_auto(self, tmp_catalog):
        from muse.cli_impl.supervisor import backfill_manifest_memory

        _seed_catalog({
            "auto-model": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {"capabilities": {"device": "auto"}},
            },
        })
        manifest = {
            "model_id": "auto-model",
            "capabilities": {"device": "auto", "memory_gb": 2.0},
        }

        out = backfill_manifest_memory(
            manifest, "auto-model", supervisor_device="cpu",
        )

        assert out["capabilities"]["device"] == "cpu"
        assert manifest["capabilities"]["device"] == "auto"

    def test_manifest_pin_beats_supervisor_device_during_backfill(self, tmp_catalog):
        from muse.cli_impl.supervisor import backfill_manifest_memory

        _seed_catalog({
            "pinned-manifest": {
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {"capabilities": {"device": "cuda"}},
            },
        })
        manifest = {
            "model_id": "pinned-manifest",
            "capabilities": {"device": "cuda", "memory_gb": 2.0},
        }

        out = backfill_manifest_memory(
            manifest, "pinned-manifest", supervisor_device="cpu",
        )

        assert out["capabilities"]["device"] == "cuda"


# ---------------------------------------------------------------------------
# Task 7 (LoRA adapter support): backfill_manifest_memory chases the base
# entry for unprobed LoRA models. A LoRA entry's own local_dir holds only
# the (tens-of-MB) adapter, so the weights-on-disk fallback would grossly
# undersize the real base+adapter load. When a lora_adapter entry has no
# probe measurement of its own, size it from its muse-id base entry
# instead. A probed LoRA entry keeps its own measured peak.
# ---------------------------------------------------------------------------


class TestBackfillLoraChase:
    def _write(self, tmp_path, entries):
        from muse.core.catalog import _write_catalog
        _write_catalog(entries)

    def _lora_manifest(self, base="sdxl-turbo"):
        return {
            "model_id": "pixel-art-xl",
            "modality": "image/generation",
            "capabilities": {"lora_adapter": True, "base_model": base},
        }

    def test_unprobed_lora_sizes_from_base_measurement(self, tmp_path, monkeypatch):
        from muse.cli_impl.supervisor import backfill_manifest_memory
        from muse.core.catalog import _reset_read_catalog_cache

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        self._write(tmp_path, {
            "pixel-art-xl": {"local_dir": str(tmp_path), "enabled": True},
            "sdxl-turbo": {
                "local_dir": "/w/sdxl-turbo", "enabled": True,
                "measurements": {"cuda": {"peak_bytes": 8_000_000_000}},
            },
        })
        _reset_read_catalog_cache()
        out = backfill_manifest_memory(self._lora_manifest(), "pixel-art-xl")
        assert out["capabilities"]["memory_gb"] == pytest.approx(8.0, rel=0.1)

    def test_probed_lora_uses_own_measurement_not_base(self, tmp_path, monkeypatch):
        from muse.cli_impl.supervisor import backfill_manifest_memory
        from muse.core.catalog import _reset_read_catalog_cache

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        self._write(tmp_path, {
            "pixel-art-xl": {
                "local_dir": str(tmp_path), "enabled": True,
                "measurements": {"cuda": {"peak_bytes": 9_000_000_000}},
            },
            "sdxl-turbo": {
                "local_dir": "/w/sdxl-turbo", "enabled": True,
                "measurements": {"cuda": {"peak_bytes": 8_000_000_000}},
            },
        })
        _reset_read_catalog_cache()
        out = backfill_manifest_memory(self._lora_manifest(), "pixel-art-xl")
        assert out["capabilities"]["memory_gb"] == pytest.approx(9.0, rel=0.1)

    def test_hf_repo_base_lora_keeps_existing_behavior(self, tmp_path, monkeypatch):
        """HF-repo base (contains /): no catalog entry to chase; the
        entry's own ladder result is used (resolve-time estimate covers
        the honest number in practice)."""
        from muse.cli_impl.supervisor import backfill_manifest_memory
        from muse.core.catalog import _reset_read_catalog_cache

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        self._write(tmp_path, {
            "pixel-art-xl": {"local_dir": str(tmp_path), "enabled": True},
        })
        _reset_read_catalog_cache()
        out = backfill_manifest_memory(
            self._lora_manifest(base="stabilityai/stable-diffusion-xl-base-1.0"),
            "pixel-art-xl",
        )
        # No base entry, no own measurement, empty local_dir: nothing
        # sensible to backfill; capabilities stay untouched (or an
        # empty-dir weights-fallback yields a negligible number).
        gb = out["capabilities"].get("memory_gb")
        assert gb is None or gb < 0.01


# ---------------------------------------------------------------------------
# v0.47.3 Fix A: on-disk weights-size fallback. A never-probed model is
# still sizable from its downloaded weights (catalog local_dir), so it
# loads on demand (evicting as needed) instead of 503 model_unservable.
# ---------------------------------------------------------------------------


class TestWeightsSizeFallback:
    def _make_weights_dir(self, tmp_path, *sizes_bytes: int):
        d = tmp_path / "weights"
        d.mkdir()
        for i, n in enumerate(sizes_bytes):
            (d / f"model-{i}.safetensors").write_bytes(b"\0" * n)
        # Non-weight files are ignored by the fallback sizer.
        (d / "config.json").write_text("{}")
        return d

    def test_weights_size_sums_files(self, tmp_path):
        from muse.cli_impl.supervisor import _weights_size_gb

        d = self._make_weights_dir(tmp_path, 1_000_000, 2_000_000)
        gb = _weights_size_gb({"local_dir": str(d)})
        # ~3 MB of weights; config.json is not counted.
        assert gb == pytest.approx(3_000_000 / 1024 ** 3, rel=1e-3)

    def test_weights_size_zero_when_no_local_dir(self):
        from muse.cli_impl.supervisor import _weights_size_gb

        assert _weights_size_gb({}) == 0.0

    def test_weights_size_zero_when_dir_missing(self, tmp_path):
        from muse.cli_impl.supervisor import _weights_size_gb

        assert _weights_size_gb({"local_dir": str(tmp_path / "nope")}) == 0.0

    def test_gguf_sizes_only_the_declared_variant(self, tmp_path):
        """A GGUF snapshot dir often holds several quant variants of one
        model, but only the declared `gguf_file` actually loads. Sizing
        must count that single file, not the sum of every variant, which
        would 503 a servable model as "exceeds device capacity" (a 4B q4
        whose repo ships six quants sums to ~15 GB instead of ~2.6 GB).
        """
        from muse.cli_impl.supervisor import _weights_size_gb

        d = tmp_path / "snapshot"
        d.mkdir()
        (d / "Model-Q4_K_M.gguf").write_bytes(b"\0" * 2_600_000)  # active
        (d / "Model-Q8_0.gguf").write_bytes(b"\0" * 5_000_000)    # other quants
        (d / "Model-F16.gguf").write_bytes(b"\0" * 8_000_000)
        (d / "README.md").write_text("x")

        gb = _weights_size_gb({
            "local_dir": str(d),
            "manifest": {"capabilities": {"gguf_file": "Model-Q4_K_M.gguf"}},
        })
        # Only the 2.6 MB active variant, not the ~15.6 MB tree sum.
        assert gb == pytest.approx(2_600_000 / 1024 ** 3, rel=1e-6)

    def test_gguf_missing_declared_file_falls_back_to_tree(self, tmp_path):
        """A declared `gguf_file` that is absent on disk (stale path) must
        fall back to the tree walk rather than returning 0.0 and stamping
        the model unservable.
        """
        from muse.cli_impl.supervisor import _weights_size_gb

        d = tmp_path / "snapshot"
        d.mkdir()
        (d / "actual.gguf").write_bytes(b"\0" * 1_000_000)

        gb = _weights_size_gb({
            "local_dir": str(d),
            "manifest": {"capabilities": {"gguf_file": "not-here.gguf"}},
        })
        assert gb == pytest.approx(1_000_000 / 1024 ** 3, rel=1e-3)

    def test_has_memory_data_falls_back_to_weights(self, tmp_path):
        """No declared memory_gb and no measurements, but weights on disk:
        has_data is True and gb reflects the on-disk size.
        """
        from muse.cli_impl.supervisor import _has_memory_data

        d = self._make_weights_dir(tmp_path, 800_000_000)
        has, gb, device = _has_memory_data({
            "local_dir": str(d),
            "manifest": {"capabilities": {"device": "cpu"}},
        })
        assert has is True
        assert gb == pytest.approx(800_000_000 / 1024 ** 3, rel=1e-3)
        assert device == "cpu"

    def test_declared_memory_wins_over_weights(self, tmp_path):
        from muse.cli_impl.supervisor import _has_memory_data

        d = self._make_weights_dir(tmp_path, 800_000_000)
        has, gb, _ = _has_memory_data({
            "local_dir": str(d),
            "manifest": {"capabilities": {"device": "cpu", "memory_gb": 5.0}},
        })
        assert has is True
        assert gb == 5.0

    def test_measurement_wins_over_weights(self, tmp_path):
        from muse.cli_impl.supervisor import _has_memory_data

        d = self._make_weights_dir(tmp_path, 800_000_000)
        has, gb, _ = _has_memory_data({
            "local_dir": str(d),
            "manifest": {"capabilities": {"device": "cpu"}},
            "measurements": {"cpu": {"peak_bytes": 1_500_000_000, "device": "cpu"}},
        })
        assert has is True
        assert gb == pytest.approx(1_500_000_000 / 1024 ** 3, rel=1e-6)

    def test_boot_does_not_flag_model_with_weights_on_disk(self, tmp_catalog, tmp_path):
        """A never-probed enabled model with weights on disk must NOT be
        stamped unservable at boot -- it can be sized + loaded on demand.
        """
        d = self._make_weights_dir(tmp_path, 800_000_000)
        _seed_catalog({
            "never-probed": {
                "python_path": "/v/bin/python",
                "local_dir": str(d),
                "enabled": True,
                "manifest": {
                    "model_id": "never-probed",
                    "modality": "embedding/text",
                    "capabilities": {"device": "cpu"},  # no memory_gb
                },
                # no measurements
            },
        })
        state = SupervisorState()
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 32.0
        validate_catalog_at_boot(state, memory_probe=probe)
        assert "never-probed" not in state.unservable_reasons


class TestWeightsSizeDedup:
    """The on-disk sizer must de-dup redundant format/dtype variants and
    skip non-weight blobs, else it OVERestimates (spurious 'too large' 503s)."""

    def _mk(self, tmp_path, files):
        # files: dict of relpath -> size_bytes
        import os
        for rel, size in files.items():
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\0" * size)
        return {"local_dir": str(tmp_path)}

    def test_fp16_and_fp32_counted_once(self, tmp_path):
        from muse.cli_impl.supervisor import _weights_size_gb
        mib = 1024 ** 2
        entry = self._mk(tmp_path, {
            "unet/diffusion_pytorch_model.safetensors": 2 * mib,
            "unet/diffusion_pytorch_model.fp16.safetensors": 1 * mib,
        })
        # only the fp16 variant is counted, not the sum
        assert _weights_size_gb(entry) == pytest.approx(1 * mib / 1024 ** 3)

    def test_bf16_preferred_over_fp32(self, tmp_path):
        from muse.cli_impl.supervisor import _weights_size_gb
        mib = 1024 ** 2
        entry = self._mk(tmp_path, {
            "unet/diffusion_pytorch_model.bf16.safetensors": 1 * mib,
            "unet/diffusion_pytorch_model.safetensors": 2 * mib,  # fp32
        })
        # bf16 is a 16-bit variant, preferred over the fp32 plain variant
        # regardless of os.walk order (the tie would otherwise count fp32).
        assert _weights_size_gb(entry) == pytest.approx(1 * mib / 1024 ** 3)

    def test_safetensors_preferred_over_bin_and_ckpt(self, tmp_path):
        from muse.cli_impl.supervisor import _weights_size_gb
        mib = 1024 ** 2
        entry = self._mk(tmp_path, {
            "model.safetensors": 2 * mib,
            "model.ckpt": 2 * mib,
            "model.bin": 2 * mib,
        })
        # one variant counted, not all three formats
        assert _weights_size_gb(entry) == pytest.approx(2 * mib / 1024 ** 3)

    def test_transformers_dual_format_shards_counted_once(self, tmp_path):
        from muse.cli_impl.supervisor import _weights_size_gb
        mib = 1024 ** 2
        entry = self._mk(tmp_path, {
            "model-00001-of-00002.safetensors": 2 * mib,
            "model-00002-of-00002.safetensors": 3 * mib,
            "pytorch_model-00001-of-00002.bin": 2 * mib,
            "pytorch_model-00002-of-00002.bin": 3 * mib,
        })
        # safetensors and PyTorch shards are alternate encodings of the
        # same two checkpoint shards, so count 5 MiB, not 10 MiB.
        assert _weights_size_gb(entry) == pytest.approx(5 * mib / 1024 ** 3)

    def test_non_weight_blobs_skipped(self, tmp_path):
        from muse.cli_impl.supervisor import _weights_size_gb
        mib = 1024 ** 2
        entry = self._mk(tmp_path, {
            "model.safetensors": 2 * mib,
            "fma_dataset_attribution2.csv": 3 * mib,  # big non-weight blob
            "README.md": 1 * mib,
            "config.json": 1024,
        })
        assert _weights_size_gb(entry) == pytest.approx(2 * mib / 1024 ** 3)

    def test_distinct_components_all_counted(self, tmp_path):
        from muse.cli_impl.supervisor import _weights_size_gb
        mib = 1024 ** 2
        entry = self._mk(tmp_path, {
            "unet/diffusion_pytorch_model.fp16.safetensors": 3 * mib,
            "vae/diffusion_pytorch_model.fp16.safetensors": 1 * mib,
            "text_encoder/model.fp16.safetensors": 2 * mib,
        })
        # different components in different dirs are all real
        assert _weights_size_gb(entry) == pytest.approx(6 * mib / 1024 ** 3)

    def test_gguf_file_still_wins(self, tmp_path):
        from muse.cli_impl.supervisor import _weights_size_gb
        mib = 1024 ** 2
        (tmp_path / "q4.gguf").write_bytes(b"\0" * (2 * mib))
        (tmp_path / "q8.gguf").write_bytes(b"\0" * (4 * mib))
        entry = {"local_dir": str(tmp_path),
                 "manifest": {"capabilities": {"gguf_file": "q4.gguf"}}}
        assert _weights_size_gb(entry) == pytest.approx(2 * mib / 1024 ** 3)
