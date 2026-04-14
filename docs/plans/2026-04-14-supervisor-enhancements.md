# Supervisor Enhancements: Enable/Disable + Auto-Restart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two supervisor-level features that turn `muse serve` into a long-running daemon worth depending on: (1) `muse models enable/disable` so users control which pulled models the supervisor loads without editing catalog.json by hand; (2) automatic worker restart with exponential backoff + budget so a crashed worker self-recovers instead of silently disappearing from `/v1/models`.

**Architecture:** Enable/disable is a catalog field (`enabled: bool`, defaults True), filtered at `plan_workers`. Auto-restart is a daemon thread spawned from `run_supervisor` before `uvicorn.run`, polling each worker via `/health` and `Popen.poll()`. Per-worker state (`restart_count`, `failure_count`, `status`) lives on the `WorkerSpec` dataclass. Monitor thread respects a `threading.Event` for coordinated shutdown. Gateway route table is never mutated: restarts reuse the same port, and permanently-dead workers contribute 503-equivalent errors via the existing `/v1/models` + `/health` aggregation (already tolerates worker failures by design).

**Tech Stack:** stdlib `threading` + `subprocess` (no new deps), httpx for health checks (already a server extra), pytest for tests. Python 3.10+.

---

## File Structure (final)

```
src/muse/core/catalog.py              MODIFIED: pull() sets enabled=True; _read_catalog() backfills enabled for legacy entries
src/muse/cli.py                       MODIFIED: add `muse models enable` and `muse models disable` subcommands
src/muse/cli_impl/supervisor.py       MODIFIED: WorkerSpec gains device + restart_count + failure_count + last_spawn_at + status fields
                                      MODIFIED: plan_workers skips disabled entries
                                      NEW: _check_worker_health, _attempt_restart, _monitor_workers
                                      MODIFIED: run_supervisor starts monitor thread, stops it in finally

tests/core/test_catalog.py            MODIFIED: test enabled field defaults + migration
tests/cli_impl/test_supervisor.py     MODIFIED: existing plan_workers tests + disabled-filter test
                                      NEW: TestWorkerSpec, TestCheckWorkerHealth, TestAttemptRestart, TestMonitorLoop
tests/test_cli.py                     MODIFIED: test enable/disable subcommands parse

README.md, CLAUDE.md                  MODIFIED: describe enable/disable + auto-restart semantics
```

---

## Key design decisions

1. **`enabled` defaults True on pull** — preserves current behavior for users who never call enable/disable. Catalog entries from before this change are treated as enabled via `.get("enabled", True)` in the migration path.

2. **Monitor thread, not async task** — the supervisor isn't itself an async program; the gateway FastAPI app runs under uvicorn's event loop, which is a separate thread tree. Spawning a plain `threading.Thread(daemon=True)` from the supervisor process (before `uvicorn.run`) is simpler than plumbing the monitor through FastAPI's `lifespan` context.

3. **`threading.Event` coordinates shutdown** — the monitor's sleep loop is `event.wait(interval)`, which returns early when the event is set. `run_supervisor`'s `finally` block sets the event before calling `_shutdown_workers`, so no race between restart-in-progress and shutdown.

4. **Port reuse is acceptable** — when a worker dies and we respawn it, we reuse the same port. Linux `bind()` can briefly fail with `EADDRINUSE` during TIME_WAIT; our `spawn_worker` + `wait_for_ready` sequence handles this by waiting up to 60s for `/health` to come up. If the port is genuinely stuck, the backoff + retry will surface the issue.

5. **Restart budget is 10, interval 5s, failure threshold 3** — module constants. Not CLI-configurable in this iteration (YAGNI). Default means a worker that's been failing for ~15s consecutive triggers a restart; a worker that can't stay up through 10 restart attempts is marked dead.

6. **WorkerSpec gains `device` field** — monitor thread calls `spawn_worker(spec, device=spec.device)` on restart. The alternative (closure-capturing device in the monitor function) works but puts the device value out-of-band; putting it on the spec keeps all worker state in one dataclass.

7. **Dead worker visibility** — a permanently-dead worker stays in the route table but fails all requests to it (gateway forwards, httpx fails, returns 502-ish). `/health` shows `degraded`. `/v1/models` skips it. No schema changes needed in the gateway.

8. **Disabled models don't become workers** — `plan_workers` filters `enabled == False` entries before grouping. Users who want a model "temporarily off" don't need to uninstall it; `muse models disable <id>` is a bitflip.

---

## Task graph

```
A (enable/disable field) -> B (enable/disable CLI) -> C (plan_workers filter)
                                                    \
D (WorkerSpec extensions) -> E (health check pure helpers) -> F (monitor thread + restart logic) -> G (run_supervisor integration)
                                                                                                   \
                                                                    H (docs + e2e smoke) -> I (verify + merge)
```

A → B → C is the enable/disable slice. D → E → F → G is the auto-restart slice. Both slices are independent until H.

9 tasks total.

---

## Part A — Enable/Disable Field

### Task A1: `enabled` field in catalog

**Files:**
- Modify: `src/muse/core/catalog.py` (pull + _read_catalog)
- Modify: `tests/core/test_catalog.py` (new + updated tests)

- [ ] **Step 1: Append failing tests to `tests/core/test_catalog.py`**

```python
def test_pull_records_enabled_true_by_default(tmp_catalog):
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    catalog = _read_catalog()
    assert catalog["soprano-80m"]["enabled"] is True


def test_read_catalog_backfills_enabled_for_legacy_entries(tmp_catalog):
    """Old catalog.json entries without `enabled` are treated as enabled.

    This is the migration path: no destructive writes, just a default
    when reading. Existing entries stay valid after the schema change.
    """
    import json
    from muse.core.catalog import _catalog_path
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write a legacy entry (no `enabled` field)
    p.write_text(json.dumps({
        "legacy-model": {
            "pulled_at": "...",
            "hf_repo": "x",
            "local_dir": "/x",
            "venv_path": "/v",
            "python_path": "/v/bin/python",
        },
    }))

    catalog = _read_catalog()
    assert catalog["legacy-model"]["enabled"] is True


def test_is_enabled_helper_returns_true_for_entry_with_flag(tmp_catalog):
    from muse.core.catalog import is_enabled
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    assert is_enabled("soprano-80m") is True


def test_is_enabled_helper_returns_false_after_set_enabled_false(tmp_catalog):
    from muse.core.catalog import is_enabled, set_enabled
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    set_enabled("soprano-80m", False)
    assert is_enabled("soprano-80m") is False


def test_set_enabled_raises_on_unknown_model(tmp_catalog):
    from muse.core.catalog import set_enabled
    with pytest.raises(KeyError, match="not pulled"):
        set_enabled("not-pulled-model", True)


def test_set_enabled_preserves_other_fields(tmp_catalog):
    from muse.core.catalog import set_enabled
    with patch("muse.core.catalog.create_venv"), \
         patch("muse.core.catalog.install_into_venv"), \
         patch("muse.core.catalog.snapshot_download", return_value="/fake"), \
         patch("muse.core.catalog.check_system_packages", return_value=[]):
        pull("soprano-80m")
    before = _read_catalog()["soprano-80m"]
    set_enabled("soprano-80m", False)
    after = _read_catalog()["soprano-80m"]
    # Everything except `enabled` is preserved
    for key in ("pulled_at", "hf_repo", "local_dir", "venv_path", "python_path"):
        assert before[key] == after[key]
    assert after["enabled"] is False
```

- [ ] **Step 2: Run — verify all fail**

```bash
cd /home/spinoza/github/repos/muse
pytest tests/core/test_catalog.py::test_pull_records_enabled_true_by_default \
       tests/core/test_catalog.py::test_read_catalog_backfills_enabled_for_legacy_entries \
       tests/core/test_catalog.py::test_is_enabled_helper_returns_true_for_entry_with_flag \
       tests/core/test_catalog.py::test_is_enabled_helper_returns_false_after_set_enabled_false \
       tests/core/test_catalog.py::test_set_enabled_raises_on_unknown_model \
       tests/core/test_catalog.py::test_set_enabled_preserves_other_fields -v
```

Expected: 6 fails. `is_enabled` and `set_enabled` don't exist; `enabled` field isn't written.

- [ ] **Step 3: Modify `pull()` in `src/muse/core/catalog.py`**

Find the line in `pull()` that writes the catalog entry. Currently:

```python
    catalog[model_id] = {
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "hf_repo": entry.hf_repo,
        "local_dir": str(local_dir),
        "venv_path": str(venv_path),
        "python_path": str(venv_python(venv_path)),
    }
```

Change to (add `enabled: True`):

```python
    catalog[model_id] = {
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "hf_repo": entry.hf_repo,
        "local_dir": str(local_dir),
        "venv_path": str(venv_path),
        "python_path": str(venv_python(venv_path)),
        "enabled": True,
    }
```

- [ ] **Step 4: Update `_read_catalog()` to backfill `enabled=True` for legacy entries**

Find `_read_catalog()`. Currently:

```python
def _read_catalog() -> dict:
    p = _catalog_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        logger.warning("catalog at %s corrupt; resetting", p)
        return {}
```

Change to:

```python
def _read_catalog() -> dict:
    p = _catalog_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        logger.warning("catalog at %s corrupt; resetting", p)
        return {}
    # Backfill enabled=True for pre-enable-flag entries (migration path)
    for entry in data.values():
        entry.setdefault("enabled", True)
    return data
```

- [ ] **Step 5: Add `is_enabled` and `set_enabled` helpers**

Append to `src/muse/core/catalog.py` (after `remove()`):

```python
def is_enabled(model_id: str) -> bool:
    """Return True if model is pulled AND enabled in the catalog."""
    catalog = _read_catalog()
    if model_id not in catalog:
        return False
    return catalog[model_id].get("enabled", True)


def set_enabled(model_id: str, enabled: bool) -> None:
    """Toggle the `enabled` flag for a pulled model.

    Raises KeyError if the model is not in the catalog (not pulled).
    Other catalog fields are preserved.
    """
    catalog = _read_catalog()
    if model_id not in catalog:
        raise KeyError(f"model {model_id!r} is not pulled")
    catalog[model_id]["enabled"] = bool(enabled)
    _write_catalog(catalog)
```

- [ ] **Step 6: Run — pass**

```bash
pytest tests/core/test_catalog.py -v
```

Expected: all catalog tests pass (previous + 6 new).

- [ ] **Step 7: Full regression**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

Expected: 329 passed (323 + 6 new).

- [ ] **Step 8: Commit**

```bash
git add src/muse/core/catalog.py tests/core/test_catalog.py
git commit -m "feat(core): add enabled field to catalog + is_enabled / set_enabled

pull() writes enabled=True by default. _read_catalog backfills
enabled=True for pre-flag entries so the schema change is non-
destructive (no migration required; existing catalog.json stays
valid). set_enabled toggles the field, is_enabled reads it.

Supervisor filters disabled models in Task A3."
```

---

### Task A2: `muse models enable/disable` CLI commands

**Files:**
- Modify: `src/muse/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing tests to `tests/test_cli.py`**

Append to the end:

```python
def test_models_enable_subcommand_parses():
    r = _run("models", "enable", "soprano-80m", "--help")
    assert r.returncode == 0
    # --help short-circuits; we just verify the subcommand exists
    combined = r.stdout + r.stderr
    assert "enable" in combined.lower()


def test_models_disable_subcommand_parses():
    r = _run("models", "disable", "soprano-80m", "--help")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "disable" in combined.lower()


def test_models_enable_unknown_model_nonzero_exit():
    """enable on a non-pulled model should nonzero with a clear message."""
    r = _run("models", "enable", "bogus-model-xyz")
    assert r.returncode != 0
    combined = (r.stdout + r.stderr).lower()
    assert "not pulled" in combined or "error" in combined


def test_models_disable_unknown_model_nonzero_exit():
    r = _run("models", "disable", "bogus-model-xyz")
    assert r.returncode != 0


def test_models_list_shows_enabled_status():
    """List output should include enabled/disabled column for pulled models."""
    r = _run("models", "list")
    assert r.returncode == 0
    # At least one known model should appear. The exact format of enabled
    # status depends on whether any model is actually pulled; for unpulled
    # models we don't enforce a specific column. For now, assert the list
    # runs cleanly and includes a known model.
    combined = r.stdout + r.stderr
    assert "soprano-80m" in combined
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/test_cli.py -k "enable or disable" -v
```

Expected: FAIL (subcommands don't exist).

- [ ] **Step 3: Add subparsers in `src/muse/cli.py`**

Find `_add_models_subparser` or the place where `models list`, `models info`, `models remove` are registered (likely inside `build_parser`). Add two new subparsers after `sp_info`/`sp_remove` or wherever fits cleanly:

Find the current models subtree (likely something like):

```python
    sp_list = models_sub.add_parser("list", help="...")
    ...
    sp_info = models_sub.add_parser("info", help="...")
    ...
    sp_remove = models_sub.add_parser("remove", help="...")
    ...
```

Add after `sp_remove`:

```python
    sp_enable = models_sub.add_parser("enable", help="enable a pulled model for serving")
    sp_enable.add_argument("model_id")
    sp_enable.set_defaults(func=_cmd_models_enable)

    sp_disable = models_sub.add_parser("disable", help="disable a pulled model (still in catalog, but not loaded by muse serve)")
    sp_disable.add_argument("model_id")
    sp_disable.set_defaults(func=_cmd_models_disable)
```

- [ ] **Step 4: Add handler functions**

Append to `src/muse/cli.py` (next to `_cmd_models_remove` or wherever other `_cmd_models_*` live):

```python
def _cmd_models_enable(args):
    from muse.core.catalog import set_enabled
    try:
        set_enabled(args.model_id, True)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"enabled {args.model_id}")
    return 0


def _cmd_models_disable(args):
    from muse.core.catalog import set_enabled
    try:
        set_enabled(args.model_id, False)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"disabled {args.model_id}")
    return 0
```

- [ ] **Step 5: Update `_cmd_models_list` to show enabled status**

Find `_cmd_models_list` in `cli.py`. Current loop body is likely:

```python
    for e in entries:
        status = "pulled" if is_pulled(e.model_id) else "available"
        print(f"  {e.model_id:20s} [{status:9s}] {e.modality:22s} {e.description}")
```

Change to include enabled status for pulled models:

```python
    for e in entries:
        if is_pulled(e.model_id):
            from muse.core.catalog import is_enabled
            enabled = is_enabled(e.model_id)
            status = "pulled" if enabled else "disabled"
        else:
            status = "available"
        print(f"  {e.model_id:20s} [{status:9s}] {e.modality:22s} {e.description}")
```

This shows `pulled` for enabled-and-pulled, `disabled` for disabled, `available` for not-pulled.

- [ ] **Step 6: Run — pass**

```bash
pytest tests/test_cli.py -k "enable or disable or list_shows" -v
```

Expected: 5 pass.

- [ ] **Step 7: Full regression**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

Expected: 334 passed (329 + 5 new).

- [ ] **Step 8: Commit**

```bash
git add src/muse/cli.py tests/test_cli.py
git commit -m "feat(cli): add muse models enable / disable + update list output

enable/disable toggle the catalog's enabled field via set_enabled.
Unknown-model errors exit with code 2 and a clear message. The
models list output now shows 'disabled' for pulled-but-disabled
entries so users can see what's active at a glance.

Supervisor filter in Task A3."
```

---

### Task A3: `plan_workers` skips disabled models

**Files:**
- Modify: `src/muse/cli_impl/supervisor.py`
- Modify: `tests/cli_impl/test_supervisor.py`

- [ ] **Step 1: Append failing test to `tests/cli_impl/test_supervisor.py`**

In the `TestPlanWorkers` class:

```python
    def test_skips_disabled_models(self, tmp_catalog):
        """Disabled models are filtered out of plan_workers results."""
        _seed_catalog({
            "enabled-model": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
            },
            "disabled-model": {
                "pulled_at": "...", "hf_repo": "b", "local_dir": "/b",
                "venv_path": "/venvs/b",
                "python_path": "/venvs/b/bin/python",
                "enabled": False,
            },
        })
        specs = plan_workers()
        all_models = {m for s in specs for m in s.models}
        assert "enabled-model" in all_models
        assert "disabled-model" not in all_models


    def test_legacy_entries_without_enabled_field_treated_as_enabled(self, tmp_catalog):
        """Pre-flag entries (no `enabled` key) must still be planned."""
        _seed_catalog({
            "legacy-model": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                # no 'enabled' key
            },
        })
        specs = plan_workers()
        all_models = {m for s in specs for m in s.models}
        assert "legacy-model" in all_models
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/cli_impl/test_supervisor.py::TestPlanWorkers::test_skips_disabled_models \
       tests/cli_impl/test_supervisor.py::TestPlanWorkers::test_legacy_entries_without_enabled_field_treated_as_enabled -v
```

Expected: First test fails (disabled-model is included). Second test likely passes (legacy entries already work) — but check to confirm.

- [ ] **Step 3: Update `plan_workers` in `src/muse/cli_impl/supervisor.py`**

Find the `for model_id, entry in catalog.items():` loop. Currently:

```python
    for model_id, entry in catalog.items():
        python = entry.get("python_path")
        if not python:
            logger.warning(...)
            continue
        groups.setdefault(python, []).append(model_id)
```

Add an enabled check after the python_path check:

```python
    for model_id, entry in catalog.items():
        python = entry.get("python_path")
        if not python:
            logger.warning(
                "skipping pre-worker catalog entry %r - no python_path; "
                "re-run `muse pull %s` to create its venv",
                model_id, model_id,
            )
            continue
        # Default True covers legacy entries without the field (see
        # _read_catalog's backfill, which also setdefault-s True)
        if not entry.get("enabled", True):
            logger.info("skipping disabled model %r (use `muse models enable %s`)", model_id, model_id)
            continue
        groups.setdefault(python, []).append(model_id)
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/cli_impl/test_supervisor.py -v
```

Expected: all pass (previous + 2 new).

- [ ] **Step 5: Full regression**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

Expected: 336 passed.

- [ ] **Step 6: Commit**

```bash
git add src/muse/cli_impl/supervisor.py tests/cli_impl/test_supervisor.py
git commit -m "feat(supervisor): plan_workers skips disabled models

Models with enabled=False are silently skipped with an info log.
Legacy entries (no `enabled` field) are treated as enabled via
_read_catalog's backfill. This completes the enable/disable slice:
users can muse models disable <id> to stop loading a model without
uninstalling it; muse models enable <id> brings it back."
```

---

## Part B — Auto-Restart

### Task B1: WorkerSpec extensions + device field

**Files:**
- Modify: `src/muse/cli_impl/supervisor.py`
- Modify: `tests/cli_impl/test_supervisor.py`

- [ ] **Step 1: Append failing tests to `tests/cli_impl/test_supervisor.py`**

Add a new test class:

```python
class TestWorkerSpecExtensions:
    def test_worker_spec_has_device_field_with_default(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        assert spec.device == "auto"

    def test_worker_spec_accepts_explicit_device(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cuda")
        assert spec.device == "cuda"

    def test_worker_spec_default_status_is_pending(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        assert spec.status == "pending"

    def test_worker_spec_default_restart_and_failure_counts(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        assert spec.restart_count == 0
        assert spec.failure_count == 0

    def test_worker_spec_has_last_spawn_at_default(self):
        spec = WorkerSpec(models=["x"], python_path="/p", port=9001)
        assert spec.last_spawn_at == 0.0
```

- [ ] **Step 2: Run — fail**

```bash
cd /home/spinoza/github/repos/muse
pytest tests/cli_impl/test_supervisor.py::TestWorkerSpecExtensions -v
```

Expected: FAIL (new fields don't exist).

- [ ] **Step 3: Extend `WorkerSpec` in `src/muse/cli_impl/supervisor.py`**

Find the `@dataclass class WorkerSpec:` block. Current:

```python
@dataclass
class WorkerSpec:
    """Everything needed to spawn one worker subprocess."""
    models: list[str]
    python_path: str
    port: int
    # Populated after subprocess.Popen in Task E2
    process: object = field(default=None)
```

Replace with (add the new fields while preserving existing ones):

```python
@dataclass
class WorkerSpec:
    """Everything needed to spawn and supervise one worker subprocess.

    Fields mutated by the monitor thread (after startup):
      - process: replaced on restart
      - restart_count: total restarts attempted
      - failure_count: consecutive unhealthy polls
      - last_spawn_at: time of most recent spawn (for backoff timing)
      - status: "pending" before first spawn -> "running" -> "unhealthy" -> "dead"
    """
    models: list[str]
    python_path: str
    port: int
    device: str = "auto"
    process: object = field(default=None)
    restart_count: int = 0
    failure_count: int = 0
    last_spawn_at: float = 0.0
    status: str = "pending"
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/cli_impl/test_supervisor.py::TestWorkerSpecExtensions -v
```

Expected: 5 passed.

- [ ] **Step 5: Update `spawn_worker` to record `last_spawn_at` + set status**

Find `spawn_worker` in `supervisor.py`. Current:

```python
def spawn_worker(spec: WorkerSpec, *, device: str) -> None:
    """..."""
    cmd = [
        spec.python_path, "-m", "muse.cli", "_worker",
        "--host", "127.0.0.1",
        "--port", str(spec.port),
        "--device", device,
    ]
    for m in spec.models:
        cmd.extend(["--model", m])
    logger.info("spawning worker: %s", " ".join(cmd))
    spec.process = subprocess.Popen(cmd)
```

Update to also persist `device` onto the spec (so monitor can respawn with it) and record `last_spawn_at`. Final:

```python
def spawn_worker(spec: WorkerSpec, *, device: str) -> None:
    """Start a worker subprocess using its venv's Python.

    Persists `device` onto the spec so the monitor can respawn with
    the same settings. Also records last_spawn_at for backoff timing.
    """
    spec.device = device
    cmd = [
        spec.python_path, "-m", "muse.cli", "_worker",
        "--host", "127.0.0.1",
        "--port", str(spec.port),
        "--device", device,
    ]
    for m in spec.models:
        cmd.extend(["--model", m])
    logger.info("spawning worker: %s", " ".join(cmd))
    spec.process = subprocess.Popen(cmd)
    spec.last_spawn_at = time.monotonic()
```

- [ ] **Step 6: Verify the spawn_worker test still passes**

```bash
pytest tests/cli_impl/test_supervisor.py::TestSpawnWorker -v
```

Expected: existing tests still pass. The new `spec.device = device` and `spec.last_spawn_at = time.monotonic()` lines don't change the command construction.

- [ ] **Step 7: Full regression**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

Expected: 341 passed (336 + 5 new).

- [ ] **Step 8: Commit**

```bash
git add src/muse/cli_impl/supervisor.py tests/cli_impl/test_supervisor.py
git commit -m "feat(supervisor): WorkerSpec gains device/status/restart_count fields

All monitor-thread state is now on the dataclass so pure helpers can
operate on a spec without closure-captured context. spawn_worker
persists device onto the spec and records last_spawn_at at spawn time."
```

---

### Task B2: Health-check pure helpers

**Files:**
- Modify: `src/muse/cli_impl/supervisor.py`
- Modify: `tests/cli_impl/test_supervisor.py`

- [ ] **Step 1: Append failing tests to `tests/cli_impl/test_supervisor.py`**

```python
class TestCheckWorkerHealth:
    def test_returns_true_on_200(self):
        from muse.cli_impl.supervisor import check_worker_health
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert check_worker_health(port=9001) is True

    def test_returns_false_on_non_200(self):
        from muse.cli_impl.supervisor import check_worker_health
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=500)
            assert check_worker_health(port=9001) is False

    def test_returns_false_on_connection_error(self):
        from muse.cli_impl.supervisor import check_worker_health
        import httpx
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("down", request=None)
            assert check_worker_health(port=9001) is False

    def test_returns_false_on_timeout(self):
        from muse.cli_impl.supervisor import check_worker_health
        import httpx
        with patch("muse.cli_impl.supervisor.httpx.get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("slow", request=None)
            assert check_worker_health(port=9001) is False
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/cli_impl/test_supervisor.py::TestCheckWorkerHealth -v
```

Expected: `ImportError: cannot import name 'check_worker_health'`.

- [ ] **Step 3: Add `check_worker_health` to `src/muse/cli_impl/supervisor.py`**

After `wait_for_ready`, add:

```python
def check_worker_health(*, port: int, timeout: float = 2.0) -> bool:
    """Single /health poll. Returns True iff the worker responds 200.

    Swallows all httpx errors; they indicate "unhealthy" for our purposes.
    Use for the monitor thread's periodic liveness check.
    """
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
        return r.status_code == 200
    except httpx.HTTPError:
        return False
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/cli_impl/test_supervisor.py::TestCheckWorkerHealth -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/muse/cli_impl/supervisor.py tests/cli_impl/test_supervisor.py
git commit -m "feat(supervisor): add check_worker_health for monitor polls

Non-raising wrapper around httpx.get. Returns True iff 200; treats
all errors (connect / timeout / 5xx) as unhealthy. Consumed by
the monitor loop in Task B3."
```

---

### Task B3: Monitor loop + restart logic

**Files:**
- Modify: `src/muse/cli_impl/supervisor.py`
- Modify: `tests/cli_impl/test_supervisor.py`

- [ ] **Step 1: Append failing tests to `tests/cli_impl/test_supervisor.py`**

```python
class TestAttemptRestart:
    def test_respawns_after_process_death(self):
        """If process exited, terminate (no-op if dead) + respawn."""
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001, device="cpu",
        )
        spec.process = MagicMock(poll=MagicMock(return_value=1))  # already exited
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.wait_for_ready") as mock_wait:
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        mock_spawn.assert_called_once_with(spec, device="cpu")
        mock_wait.assert_called_once()
        assert spec.restart_count == 1
        assert spec.failure_count == 0
        assert spec.status == "running"

    def test_terminates_still_running_process_before_respawn(self):
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        old_process = MagicMock(poll=MagicMock(return_value=None))  # still alive
        spec.process = old_process
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready"):
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        old_process.terminate.assert_called_once()

    def test_marks_dead_after_max_restarts(self):
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(
            models=["x"], python_path="/p", port=9001, device="cpu",
            restart_count=10,  # already at budget
        )
        spec.process = MagicMock(poll=MagicMock(return_value=1))
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn:
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        mock_spawn.assert_not_called()
        assert spec.status == "dead"

    def test_spawn_failure_keeps_status_unhealthy(self):
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=1))
        stop_event = threading.Event()

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn, \
             patch("muse.cli_impl.supervisor.wait_for_ready") as mock_wait:
            mock_wait.side_effect = TimeoutError("never ready")
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=0)

        assert spec.restart_count == 1  # counter still increments
        assert spec.status != "running"  # spawn tried, but didn't become ready

    def test_respects_stop_event_during_backoff(self):
        """If stop_event is set during backoff wait, skip the restart."""
        from muse.cli_impl.supervisor import _attempt_restart
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=1))
        stop_event = threading.Event()
        stop_event.set()  # shutdown already requested

        with patch("muse.cli_impl.supervisor.spawn_worker") as mock_spawn:
            _attempt_restart(spec, stop_event=stop_event, max_restarts=10, backoff_base=1)

        # With stop_event set, we don't respawn
        mock_spawn.assert_not_called()


class TestMonitorLoop:
    def test_monitor_calls_restart_after_threshold_failures(self):
        from muse.cli_impl.supervisor import _monitor_workers
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=None))  # alive
        stop_event = threading.Event()

        # First 3 checks fail, then we stop
        health_calls = {"count": 0}
        def health_side_effect(**kwargs):
            health_calls["count"] += 1
            if health_calls["count"] >= 4:
                stop_event.set()
            return False

        with patch("muse.cli_impl.supervisor.check_worker_health", side_effect=health_side_effect), \
             patch("muse.cli_impl.supervisor._attempt_restart") as mock_restart:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        # After 3 consecutive unhealthy polls, restart should be invoked at least once
        assert mock_restart.called

    def test_monitor_resets_failure_count_on_success(self):
        from muse.cli_impl.supervisor import _monitor_workers
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=None))
        spec.failure_count = 2  # close to threshold
        stop_event = threading.Event()

        call_count = {"n": 0}
        def health_side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                stop_event.set()
            return True  # healthy

        with patch("muse.cli_impl.supervisor.check_worker_health", side_effect=health_side_effect), \
             patch("muse.cli_impl.supervisor._attempt_restart") as mock_restart:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        assert spec.failure_count == 0
        assert spec.status == "running"
        mock_restart.assert_not_called()

    def test_monitor_stops_when_event_set(self):
        from muse.cli_impl.supervisor import _monitor_workers
        import threading

        spec = WorkerSpec(models=["x"], python_path="/p", port=9001, device="cpu")
        spec.process = MagicMock(poll=MagicMock(return_value=None))
        stop_event = threading.Event()
        stop_event.set()  # already stopped

        with patch("muse.cli_impl.supervisor.check_worker_health", return_value=True) as mock_health:
            _monitor_workers(
                [spec], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        # Loop should exit immediately without any health checks
        mock_health.assert_not_called()

    def test_monitor_skips_dead_workers(self):
        from muse.cli_impl.supervisor import _monitor_workers
        import threading

        alive_spec = WorkerSpec(models=["a"], python_path="/p", port=9001, device="cpu")
        alive_spec.process = MagicMock(poll=MagicMock(return_value=None))
        dead_spec = WorkerSpec(models=["b"], python_path="/p", port=9002, device="cpu")
        dead_spec.status = "dead"

        stop_event = threading.Event()
        checked_ports = []

        def health_side_effect(**kwargs):
            checked_ports.append(kwargs["port"])
            if len(checked_ports) >= 1:
                stop_event.set()
            return True

        with patch("muse.cli_impl.supervisor.check_worker_health", side_effect=health_side_effect):
            _monitor_workers(
                [alive_spec, dead_spec], stop_event,
                interval=0.001, failure_threshold=3, max_restarts=10,
            )

        # Only the alive worker should have been polled
        assert 9001 in checked_ports
        assert 9002 not in checked_ports
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/cli_impl/test_supervisor.py::TestAttemptRestart tests/cli_impl/test_supervisor.py::TestMonitorLoop -v
```

Expected: FAIL — `_attempt_restart` and `_monitor_workers` don't exist.

- [ ] **Step 3: Add `_attempt_restart` + `_monitor_workers` to `src/muse/cli_impl/supervisor.py`**

Append after `check_worker_health`:

```python
# Monitor defaults (module constants; not CLI-configurable in this iteration)
_MONITOR_INTERVAL = 5.0
_FAILURE_THRESHOLD = 3
_MAX_RESTARTS = 10
_BACKOFF_CAP = 30.0  # seconds
_BACKOFF_BASE = 1.0


def _attempt_restart(
    spec: WorkerSpec,
    *,
    stop_event: "threading.Event",
    max_restarts: int = _MAX_RESTARTS,
    backoff_base: float = _BACKOFF_BASE,
    backoff_cap: float = _BACKOFF_CAP,
    ready_timeout: float = 60.0,
) -> None:
    """Terminate existing process if alive, wait backoff, respawn.

    Mutates spec.process, spec.restart_count, spec.failure_count, spec.status.
    Marks spec.status = "dead" if restart_count reaches max_restarts.
    Returns early if stop_event fires during backoff.
    """
    if spec.restart_count >= max_restarts:
        logger.error(
            "worker on port %d: exhausted %d restart attempts; marking dead",
            spec.port, max_restarts,
        )
        spec.status = "dead"
        return

    # Exponential backoff, capped
    backoff = min(backoff_base * (2 ** spec.restart_count), backoff_cap)
    logger.warning(
        "worker on port %d: restart attempt %d after %.1fs backoff",
        spec.port, spec.restart_count + 1, backoff,
    )
    # wait() returns True if event was set during the wait
    if stop_event.wait(backoff):
        return

    # Terminate existing process if still alive (best-effort)
    if spec.process is not None and spec.process.poll() is None:
        try:
            spec.process.terminate()
            try:
                spec.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                spec.process.kill()
        except Exception as e:
            logger.warning("worker on port %d: terminate failed: %s", spec.port, e)

    # Respawn. Always bump restart_count so we can't loop forever
    spec.restart_count += 1
    try:
        spawn_worker(spec, device=spec.device)
        wait_for_ready(port=spec.port, timeout=ready_timeout)
        spec.failure_count = 0
        spec.status = "running"
        logger.info("worker on port %d: successfully restarted", spec.port)
    except (subprocess.SubprocessError, TimeoutError) as e:
        logger.error("worker on port %d: restart failed: %s", spec.port, e)
        spec.status = "unhealthy"


def _monitor_workers(
    specs: list[WorkerSpec],
    stop_event: "threading.Event",
    *,
    interval: float = _MONITOR_INTERVAL,
    failure_threshold: int = _FAILURE_THRESHOLD,
    max_restarts: int = _MAX_RESTARTS,
) -> None:
    """Poll each worker; restart after `failure_threshold` consecutive failures.

    Exits when stop_event is set. Called from the monitor daemon thread
    started by run_supervisor.
    """
    while not stop_event.is_set():
        for spec in specs:
            if stop_event.is_set():
                return
            if spec.status == "dead":
                continue

            # Process-death detection is unambiguous; short-circuit
            if spec.process is not None and spec.process.poll() is not None:
                logger.warning(
                    "worker on port %d: process exited with code %s",
                    spec.port, spec.process.returncode,
                )
                spec.failure_count = failure_threshold
            else:
                healthy = check_worker_health(port=spec.port)
                if healthy:
                    spec.failure_count = 0
                    spec.status = "running"
                    continue
                spec.failure_count += 1
                if spec.status == "running":
                    spec.status = "unhealthy"
                logger.info(
                    "worker on port %d: unhealthy (%d/%d consecutive failures)",
                    spec.port, spec.failure_count, failure_threshold,
                )

            if spec.failure_count >= failure_threshold:
                _attempt_restart(spec, stop_event=stop_event, max_restarts=max_restarts)

        # Sleep with early-exit if stop_event fires
        if stop_event.wait(interval):
            return
```

Also add `import threading` at the top of the module if not already present (the test uses it, so we need it in the source too for the type annotation to resolve when tests import).

Check imports at top of `supervisor.py`:

```bash
head -15 src/muse/cli_impl/supervisor.py
```

If `import threading` isn't there, add it alongside `import subprocess`, `import time`, etc.

- [ ] **Step 4: Run — pass**

```bash
pytest tests/cli_impl/test_supervisor.py -v 2>&1 | tail -20
```

Expected: all pass (previous + 4 restart + 4 monitor = 8 new tests).

- [ ] **Step 5: Full regression**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

Expected: 353 passed (341 + 4 restart + 4 monitor + 4 check_worker_health from Task B2 already merged).

Note: the exact count depends on how many tests were added in each sub-step. Verify directly rather than trusting my running tally.

- [ ] **Step 6: Commit**

```bash
git add src/muse/cli_impl/supervisor.py tests/cli_impl/test_supervisor.py
git commit -m "feat(supervisor): add _monitor_workers + _attempt_restart

Monitor loop polls each spec's /health (via check_worker_health) and
also checks for process death via Popen.poll. After 3 consecutive
failures, calls _attempt_restart which terminates the existing
process (if alive), waits exponential backoff, respawns via
spawn_worker + wait_for_ready. Gives up after 10 attempts and
marks the spec dead.

stop_event lets run_supervisor coordinate shutdown with the
monitor thread (Task B4)."
```

---

### Task B4: `run_supervisor` starts/stops the monitor thread

**Files:**
- Modify: `src/muse/cli_impl/supervisor.py`
- Modify: `tests/cli_impl/test_supervisor.py`

- [ ] **Step 1: Append failing tests to `tests/cli_impl/test_supervisor.py`**

In the existing `TestRunSupervisor` class (or as additional tests):

```python
class TestRunSupervisorMonitor:
    def test_run_supervisor_starts_monitor_thread(self, tmp_catalog):
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready"), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"), \
             patch("muse.cli_impl.supervisor.threading.Thread") as mock_thread_cls:
            mock_uvicorn.run.side_effect = KeyboardInterrupt()
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

            # A daemon thread was created and started
            mock_thread_cls.assert_called_once()
            kwargs = mock_thread_cls.call_args.kwargs
            assert kwargs.get("daemon") is True
            assert kwargs.get("target") is not None
            mock_thread.start.assert_called_once()

    def test_run_supervisor_sets_stop_event_on_exit(self, tmp_catalog):
        """On shutdown path, the monitor must be told to stop."""
        _seed_catalog({
            "model-a": {
                "pulled_at": "...", "hf_repo": "a", "local_dir": "/a",
                "venv_path": "/venvs/a",
                "python_path": "/venvs/a/bin/python",
                "enabled": True,
            },
        })
        from muse.cli_impl.supervisor import run_supervisor
        import threading

        # Capture the actual Event created inside run_supervisor
        captured_events = []
        real_event_cls = threading.Event
        def capture_event(*a, **kw):
            e = real_event_cls(*a, **kw)
            captured_events.append(e)
            return e

        with patch("muse.cli_impl.supervisor.spawn_worker"), \
             patch("muse.cli_impl.supervisor.wait_for_ready"), \
             patch("muse.cli_impl.supervisor.uvicorn") as mock_uvicorn, \
             patch("muse.cli_impl.supervisor._shutdown_workers"), \
             patch("muse.cli_impl.supervisor.threading.Event", side_effect=capture_event), \
             patch("muse.cli_impl.supervisor.threading.Thread"):
            mock_uvicorn.run.side_effect = KeyboardInterrupt()
            run_supervisor(host="0.0.0.0", port=8000, device="cpu")

        # The shutdown Event was set
        assert captured_events, "no threading.Event was created"
        assert any(e.is_set() for e in captured_events)
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/cli_impl/test_supervisor.py::TestRunSupervisorMonitor -v
```

Expected: FAIL (monitor thread not started in run_supervisor).

- [ ] **Step 3: Wire monitor into `run_supervisor`**

Find `run_supervisor` in `supervisor.py`. Current (after E3):

```python
def run_supervisor(*, host: str, port: int, device: str) -> int:
    """..."""
    from muse.cli_impl.gateway import WorkerRoute, build_gateway

    specs = plan_workers()
    if not specs:
        logger.warning(...)

    try:
        for spec in specs:
            spawn_worker(spec, device=device)

        for spec in specs:
            logger.info(...)
            wait_for_ready(port=spec.port)

        routes = ...
        app = build_gateway(routes)

        logger.info(...)
        uvicorn.run(app, host=host, port=port, log_config=None)
    except KeyboardInterrupt:
        logger.info("shutting down (SIGINT)")
    finally:
        _shutdown_workers(specs)
    return 0
```

Add threading import if not present at top:

```python
import threading
```

Replace `run_supervisor` body with:

```python
def run_supervisor(*, host: str, port: int, device: str) -> int:
    """Entry point for `muse serve`.

    Plans workers from catalog, spawns them, waits for ready, then starts
    the monitor thread (auto-restart) + gateway. Guarantees clean
    shutdown of workers and monitor on exit.
    """
    from muse.cli_impl.gateway import WorkerRoute, build_gateway

    specs = plan_workers()
    if not specs:
        logger.warning(
            "no pulled models with a venv - server will start empty. "
            "Pull a model first: `muse pull <model-id>`"
        )

    stop_event = threading.Event()
    monitor_thread: threading.Thread | None = None

    try:
        for spec in specs:
            spawn_worker(spec, device=device)

        for spec in specs:
            logger.info("waiting for worker on port %d (%s)", spec.port, spec.models)
            wait_for_ready(port=spec.port)
            spec.status = "running"

        # Start the auto-restart monitor AFTER all workers are ready, so
        # the initial wait_for_ready isn't racing with the monitor's own
        # readiness tracking.
        if specs:
            monitor_thread = threading.Thread(
                target=_monitor_workers,
                args=(specs, stop_event),
                daemon=True,
                name="muse-monitor",
            )
            monitor_thread.start()
            logger.info("auto-restart monitor running (interval=%.1fs, threshold=%d, budget=%d)",
                        _MONITOR_INTERVAL, _FAILURE_THRESHOLD, _MAX_RESTARTS)

        routes: list[WorkerRoute] = []
        for spec in specs:
            worker_url = f"http://127.0.0.1:{spec.port}"
            for m in spec.models:
                routes.append(WorkerRoute(model_id=m, worker_url=worker_url))
        app = build_gateway(routes)

        logger.info("starting gateway on %s:%d", host, port)
        uvicorn.run(app, host=host, port=port, log_config=None)
    except KeyboardInterrupt:
        logger.info("shutting down (SIGINT)")
    finally:
        # Tell the monitor to stop BEFORE shutting down workers.
        # Otherwise the monitor could spawn a restart while we're
        # terminating processes.
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5.0)
        _shutdown_workers(specs)
    return 0
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/cli_impl/test_supervisor.py -v 2>&1 | tail -20
```

Expected: all pass including the 2 new `TestRunSupervisorMonitor` tests.

- [ ] **Step 5: Full regression + e2e**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
pytest tests/cli_impl/test_e2e_supervisor.py -v --timeout=60
```

Expected: unit suite passes; e2e test still passes (subprocess-spawning works with monitor thread now running).

- [ ] **Step 6: Commit**

```bash
git add src/muse/cli_impl/supervisor.py tests/cli_impl/test_supervisor.py
git commit -m "feat(supervisor): run_supervisor starts auto-restart monitor thread

Daemon thread started AFTER all initial wait_for_ready calls succeed
so the monitor doesn't race with startup. stop_event set in finally
block before _shutdown_workers, and monitor_thread.join(timeout=5)
gives the monitor a chance to exit cleanly before workers get
SIGTERM. If the monitor was in the middle of a restart backoff when
shutdown fired, it exits early via the stop_event.wait() return."
```

---

## Part C — Polish + Verification

### Task C1: Docs update

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

No em-dashes (U+2014). The soul-voice hook blocks them.

- [ ] **Step 1: Update `CLAUDE.md`**

In the "Process model" section (added by the multi-venv plan), append a paragraph after the existing description:

```markdown
The supervisor also runs an auto-restart monitor thread. Every 5
seconds it polls each worker's /health and checks for process death
via Popen.poll. After 3 consecutive failures (or immediate process
exit), the monitor terminates the existing process and respawns it
with exponential backoff (1s, 2s, 4s, ..., capped at 30s). After 10
unsuccessful restart attempts the worker is marked dead; /health
reports "degraded" and /v1/models skips its entries.

Use `muse models disable <id>` to mark a pulled model as inactive
(supervisor skips it at plan_workers time, freeing its venv's memory
budget). `muse models enable <id>` re-enables it. Neither command
restarts the server; the change takes effect next `muse serve`.
```

Also in the "Project-specific conventions" list (or wherever operational notes live), add:

```markdown
- **Auto-restart is always on.** No --no-autorestart flag in this iteration. Workers that can't stay up through 10 restart attempts are marked dead; manual restart via `Ctrl+C` + `muse serve` is required to reset the counter.
- **Enable/disable is catalog state**, not runtime state. `muse serve` reads the catalog at startup. Changing a model's enabled bit while the server is running has no effect until the next restart.
```

- [ ] **Step 2: Update `README.md`**

In the "CLI (admin-only)" table, add two rows after `muse models remove`:

```markdown
| `muse models enable <model-id>` | mark a pulled model active (load on next serve) |
| `muse models disable <model-id>` | mark a pulled model inactive (skip on next serve) |
```

In the Quick start section, add a short note about restart behavior (optional but user-friendly):

```markdown
`muse serve` auto-restarts crashed worker processes with exponential backoff.
Individual model failures don't take down the server or other modalities.
```

- [ ] **Step 3: No em-dashes**

```bash
grep -n "—" CLAUDE.md README.md
```

Expected: no output. If any, replace with colons / commas / periods.

- [ ] **Step 4: Tests unaffected**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

Expected: unchanged pass count.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document enable/disable + auto-restart semantics

CLAUDE.md Process model section gains paragraphs on the monitor
thread (poll interval, failure threshold, backoff, budget) and on
enable/disable being catalog state (takes effect next serve).
README CLI table adds enable/disable rows."
```

---

### Task C2: Final verification + merge

**Files:** none (verification only)

- [ ] **Step 1: Fresh install**

```bash
cd /home/spinoza/github/repos/muse
pip install -e ".[dev,server]"
```

- [ ] **Step 2: Full test suite excluding slow**

```bash
pytest tests/ -q -m "not slow" --cov=muse --cov-report=term-missing 2>&1 | tail -30
```

Expected: everything passes. Note new coverage for `supervisor.py` (should be higher now that monitor + restart are covered).

- [ ] **Step 3: E2E slow test**

```bash
pytest tests/cli_impl/test_e2e_supervisor.py -v --timeout=60
```

Expected: 1 passed. The e2e test spawns a real subprocess; the monitor thread now runs alongside it. Must still complete.

- [ ] **Step 4: CLI smokes**

```bash
muse models enable --help
muse models disable --help
muse models list              # shows enabled/disabled status for pulled models
```

All exit 0. The list command's format may show `[pulled   ]`, `[disabled ]`, or `[available]` per model.

- [ ] **Step 5: End-to-end disable/enable cycle (no models pulled OK)**

If no models are pulled in the test environment, the following will error at enable/disable:

```bash
muse models enable bogus-model
# Expected: exits 2 with "error: model 'bogus-model' is not pulled"
```

This is the correct failure mode. If a model IS pulled, you can do:

```bash
muse models disable <some-pulled-model>
muse models list        # shows [disabled]
muse models enable <some-pulled-model>
muse models list        # shows [pulled]
```

- [ ] **Step 6: Commit any fixes**

```bash
git status
# If changes:
git add -A
git commit -m "fix: issues found in final verification"
```

- [ ] **Step 7: Show commit history on branch**

```bash
git log --oneline main..HEAD
```

Should show ~9 commits (one per task).

- [ ] **Step 8: Merge back to main with --no-ff**

(Typically handled by `superpowers:finishing-a-development-branch` at the controller level.)

```bash
git checkout main
git merge --no-ff <feature-branch> -m "feat: supervisor enable/disable + auto-restart"
git worktree remove <worktree-path>
git branch -d <feature-branch>
```

---

## Scope notes (deferred to future plans)

- **Hot reload**: supervisor watches `catalog.json` mtime and diffs running workers vs planned workers. Spawns new, terminates removed, leaves unchanged alone. Separate plan since it reuses these primitives but adds filesystem watching complexity.
- **Restart rate-limiting across workers**: if 5 workers all fail simultaneously, we'd hammer the system with parallel restart attempts. Current design has per-worker backoff; a global throttle (e.g., max 1 restart attempt per second across all workers) is a follow-up.
- **Gateway-side dead-worker handling**: today a worker marked `status="dead"` is invisible to the gateway — requests still route to its port and fail. The fix is for the gateway to consult worker status and return a specific 503 with a "worker unavailable" message. Deferred to avoid cross-cutting changes in this plan.
- **Liveness manifest endpoint**: `/v1/workers` on the gateway exposing each worker's port, models, status, restart_count. Useful for observability but YAGNI for the single-user scenario.
- **Configurable monitor parameters**: `--monitor-interval`, `--failure-threshold`, `--max-restarts` CLI flags on `muse serve`. Trivial to add when actually needed.

## Self-review

**Spec coverage:**
- Enable/disable catalog field: ✅ A1
- CLI enable/disable + list output: ✅ A2
- Supervisor filter for disabled models: ✅ A3
- WorkerSpec extensions for monitor state: ✅ B1
- Health check helper: ✅ B2
- Monitor loop + restart logic: ✅ B3
- run_supervisor lifecycle integration: ✅ B4
- Docs: ✅ C1
- Verification + merge: ✅ C2

**Placeholder scan:** No "TODO", "implement later", or vague instructions. Every step has complete code.

**Type consistency:**
- `WorkerSpec.device` default `"auto"` — consistent with `run_worker`'s `device` param default.
- `WorkerSpec.status` values: `"pending"`, `"running"`, `"unhealthy"`, `"dead"` — all four referenced consistently across B3, B4.
- `_attempt_restart(spec, *, stop_event, max_restarts, backoff_base, backoff_cap, ready_timeout)` — signature matches all call sites and tests.
- `_monitor_workers(specs, stop_event, *, interval, failure_threshold, max_restarts)` — signature matches.
- `check_worker_health(*, port, timeout)` — keyword-only `port` and `timeout`; matches tests.
- `is_enabled(model_id)` and `set_enabled(model_id, enabled)` — names consistent between catalog.py, cli.py, and supervisor.py usages.

Plan complete. 9 tasks. TDD-first throughout. ~25 new tests.
