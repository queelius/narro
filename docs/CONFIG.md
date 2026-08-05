# muse configuration

All muse server settings live in ONE declarative registry,
`muse.core.config.SETTINGS`. Every knob is a single `Setting` row
(`key`, `env`, `type`, `default`, `group`, `help`), and env reads, config-file
reads, the `muse config` CLI, and the generated template all derive from that
one list. There is no parallel lookup table to keep in sync.

## Precedence

`config.get(key)` resolves each setting, first match wins:

1. **CLI override** -- a value passed by a command flag (e.g. `muse serve
   --device cuda` overrides `server.device`).
2. **Environment variable** -- the setting's `MUSE_*` env var, read LIVE on
   every call.
3. **Config file** -- `~/.muse/config.yaml`, parsed once and cached.
4. **Built-in default** -- the registry `default`.

`config.get` is LENIENT: an un-coercible env/file value logs one warning and
falls back to the default -- it never raises, so a typo in one setting cannot
turn every request into a 500. `muse config set` is the STRICT path: it
validates the value against the registry and refuses to write a bad one.

## The config file

- Location: `<catalog_dir>/config.yaml`, where `catalog_dir` is
  `MUSE_CATALOG_DIR` (env) or `~/.muse`. Override the whole path with the
  `MUSE_CONFIG` env var.
- Format: yaml, nested by group (`server:`, `limits:`, `paths:`, `fetch:`,
  `admin:`, `client:`).
- A missing file means "all defaults" (no error). Unknown sections/keys are
  warned about and ignored.
- Bootstrap ordering: the two keys that LOCATE the file itself
  (`paths.catalog_dir`, `paths.config_file`) resolve from env+default ONLY --
  the file cannot redirect the path that finds the file. They appear in the
  generated template commented out.

## `muse config` CLI

| Command | Effect |
|---|---|
| `muse config generate [--force]` | Write a fully-commented `config.yaml` from the registry (every setting, default, env name). Refuses to overwrite without `--force`. |
| `muse config show [--json]` | Every setting's effective value AND source (default / file / env). `admin.token` is redacted to `set` / `unset`. |
| `muse config path` | Print the resolved config-file path. |
| `muse config get <key>` | Print one effective value. |
| `muse config set <key> <value>` | Validate and atomically write one value into the file (preserves other keys). |
| `muse config unset <key>` | Remove one setting from the file so it falls back to env/default (prunes an empty group). The counterpart to `set`: there is no override value meaning "use the default", so reverting a key means removing it. |

Examples:

```bash
muse config generate                       # write ~/.muse/config.yaml
muse config set server.idle_timeout_seconds 0   # disable idle eviction
muse config set limits.upscale_max_input_side 2048
muse config show --json | jq '.[] | select(.source != "default")'
```

## Scope boundary: config.yaml vs catalog.json vs curated.yaml

- `~/.muse/config.yaml` (`muse config ...`) -- SERVER / global settings. This
  file.
- `~/.muse/catalog.json` (`muse models ...`) -- per-model, per-install state:
  the `enabled` bit, `device_override` (from `muse models set-device`), probe
  `measurements`, and the persisted manifest for pulled models.
- `src/muse/curated.yaml` (shipped package data) -- muse-blessed model
  recommendations. NOT per-deployment editable; overwritten on upgrade.

A setting can have both a global default (config.yaml) and a per-model
override (manifest capability). Idle-timeout is the canonical example:
`server.idle_timeout_seconds` is the global default;
`capabilities.idle_timeout_seconds` on a model wins for that model.

## Settings inventory

The canonical, always-current list is `muse config generate` (or
`python -c "from muse.core import config; print(config.render_template())"`).
A snapshot of the defaults:

### server (read at supervisor / worker start)

| key | env | default |
|---|---|---|
| `server.idle_sweep_interval_seconds` | `MUSE_IDLE_SWEEP_INTERVAL_SECONDS` | 30.0 |
| `server.idle_timeout_seconds` | `MUSE_DEFAULT_IDLE_TIMEOUT_SECONDS` | 600.0 (10 min; `<= 0` disables) |
| `server.shutdown_grace_seconds` | `MUSE_SHUTDOWN_GRACE_SECONDS` | null |
| `server.default_max_concurrency` | `MUSE_DEFAULT_MAX_CONCURRENCY` | 0 (unlimited) |
| `server.queue_timeout_seconds` | `MUSE_QUEUE_TIMEOUT_SECONDS` | 300.0 (5 min) |
| `server.max_queue_depth` | `MUSE_MAX_QUEUE_DEPTH` | 256 (`0` opts into unbounded) |
| `server.gpu_budget_gb` | `MUSE_GPU_BUDGET_GB` | null |
| `server.cpu_budget_gb` | `MUSE_CPU_BUDGET_GB` | null |
| `server.gpu_headroom_gb` | `MUSE_GPU_HEADROOM_GB` | 1.0 |
| `server.cpu_headroom_gb` | `MUSE_CPU_HEADROOM_GB` | 2.0 |
| `server.aggregation_timeout_seconds` | `MUSE_AGGREGATION_TIMEOUT_SECONDS` | 5.0 (per-worker httpx timeout for `/v1/models` + `/health` fan-out) |
| `server.device` | `MUSE_DEVICE` | auto (`muse serve --device` overrides) |
| `server.video_cpu_offload` | `MUSE_VIDEO_CPU_OFFLOAD` | null (global override: `model`\|`sequential`\|`off`; unset uses the per-model capability) |

### admin / client

| key | env | default |
|---|---|---|
| `admin.token` | `MUSE_ADMIN_TOKEN` | null (unset keeps admin closed) |
| `client.server_url` | `MUSE_SERVER` | http://localhost:8000 |

### paths (bootstrap: catalog_dir / config_file resolve env+default only)

| key | env | default |
|---|---|---|
| `paths.catalog_dir` | `MUSE_CATALOG_DIR` | ~/.muse |
| `paths.home` | `MUSE_HOME` | ~/.muse |
| `paths.models_dir` | `MUSE_MODELS_DIR` | null |
| `paths.modalities_dir` | `MUSE_MODALITIES_DIR` | null |
| `paths.config_file` | `MUSE_CONFIG` | null (-> catalog_dir/config.yaml) |

### fetch

| key | env | default |
|---|---|---|
| `fetch.allow_private` | `MUSE_ALLOW_PRIVATE_FETCH` | false (SSRF guard ON) |
| `fetch.mcp_allowed_path_prefixes` | `MUSE_MCP_ALLOWED_PATH_PREFIXES` | "" |

### limits (per-modality request caps)

| key | env | default |
|---|---|---|
| `limits.image_input_max_bytes` | `MUSE_IMAGE_INPUT_MAX_BYTES` | 10485760 (10 MB) |
| `limits.audio_cls_max_bytes` | `MUSE_AUDIO_CLS_MAX_BYTES` | 52428800 (50 MB) |
| `limits.audio_quality_max_bytes` | `MUSE_AUDIO_QUALITY_MAX_BYTES` | 52428800 (50 MB) |
| `limits.audio_quality_max_duration_seconds` | `MUSE_AUDIO_QUALITY_MAX_DURATION_SECONDS` | 600.0 |
| `limits.audio_alignment_max_bytes` | `MUSE_AUDIO_ALIGNMENT_MAX_BYTES` | 52428800 (50 MB) |
| `limits.audio_alignment_max_duration_seconds` | `MUSE_AUDIO_ALIGNMENT_MAX_DURATION_SECONDS` | 300.0 |
| `limits.audio_alignment_max_text_chars` | `MUSE_AUDIO_ALIGNMENT_MAX_TEXT_CHARS` | 50000 |
| `limits.audio_embeddings_max_bytes` | `MUSE_AUDIO_EMBEDDINGS_MAX_BYTES` | 52428800 (50 MB) |
| `limits.asr_max_mb` | `MUSE_ASR_MAX_MB` | 100 |
| `limits.embeddings_max_batch` | `MUSE_EMBEDDINGS_MAX_BATCH` | 2048 |
| `limits.embeddings_max_chars_per_item` | `MUSE_EMBEDDINGS_MAX_CHARS_PER_ITEM` | 100000 |
| `limits.image_embeddings_max_batch` | `MUSE_IMAGE_EMBEDDINGS_MAX_BATCH` | 64 |
| `limits.segmentation_max_input_side` | `MUSE_SEGMENTATION_MAX_INPUT_SIDE` | 2048 |
| `limits.upscale_max_input_side` | `MUSE_UPSCALE_MAX_INPUT_SIDE` | 1024 |
| `limits.vectorization_max_input_side` | `MUSE_VECTORIZATION_MAX_INPUT_SIDE` | 2048 |
| `limits.model_3d_input_max_bytes` | `MUSE_3D_INPUT_MAX_BYTES` | 20971520 (20 MB) |
| `limits.moderations_max_batch` | `MUSE_MODERATIONS_MAX_BATCH` | 1024 |
| `limits.moderations_max_chars_per_item` | `MUSE_MODERATIONS_MAX_CHARS_PER_ITEM` | 100000 |
| `limits.classifications_max_labels` | `MUSE_CLASSIFICATIONS_MAX_LABELS` | 200 |
| `limits.rerank_max_documents` | `MUSE_RERANK_MAX_DOCUMENTS` | 1000 |
| `limits.rerank_max_query_chars` | `MUSE_RERANK_MAX_QUERY_CHARS` | 4000 |
| `limits.rerank_max_doc_chars` | `MUSE_RERANK_MAX_DOC_CHARS` | 100000 |
| `limits.summarize_max_text_chars` | `MUSE_SUMMARIZE_MAX_TEXT_CHARS` | 100000 |
| `limits.translate_max_chars` | `MUSE_TRANSLATE_MAX_CHARS` | 20000 |
| `limits.video_max_frames_b64` | `MUSE_VIDEO_MAX_FRAMES_B64` | 240 |

The six `limits.*` byte/side caps that have a positivity guard
(`image_input`, `audio_cls`, `model_3d_input`, `segmentation`, `upscale`,
`vectorization`) treat
a non-positive value (`<= 0`, or empty for the opt-int byte caps) as "use the
default", so setting one to 0 falls back rather than rejecting every request.

### telemetry (observability recorder + dashboard)

| key | env | default |
|---|---|---|
| `telemetry.enabled` | `MUSE_TELEMETRY_ENABLED` | true |
| `telemetry.require_auth` | `MUSE_TELEMETRY_REQUIRE_AUTH` | true |
| `telemetry.retention_days` | `MUSE_TELEMETRY_RETENTION_DAYS` | 7 |
| `telemetry.log_buffer_kb` | `MUSE_TELEMETRY_LOG_BUFFER_KB` | 64 |
| `telemetry.sample_interval_seconds` | `MUSE_TELEMETRY_SAMPLE_INTERVAL_SECONDS` | 10.0 |
| `telemetry.trace_sample_interval_seconds` | `MUSE_TELEMETRY_TRACE_SAMPLE_INTERVAL_SECONDS` | 0.25 |
| `telemetry.log_ticket_ttl_seconds` | `MUSE_TELEMETRY_LOG_TICKET_TTL_SECONDS` | 60.0 |

`telemetry.enabled` gates the whole subsystem: the event recorder, the
per-model log ring buffer, the periodic resource sampler, and the
`/dashboard` data endpoints. Setting it to `false` swaps in a no-op
recorder (zero queue/thread overhead) and skips wiring the log hub, so
worker spawning is unchanged from a pre-telemetry `muse serve`.
`telemetry.require_auth` independently controls the admin-token gate on
dashboard data and log endpoints. It defaults to true; disable it only for an
explicitly trusted development or loopback-only deployment.
`telemetry.retention_days` bounds how far back `/v1/telemetry/series`
history is pruned. `telemetry.log_buffer_kb` sizes each model's
per-worker stdout ring buffer (KB, not lines).
`telemetry.sample_interval_seconds` is how often the background sampler
records a `sample` event (free VRAM/RAM, loaded/in-flight counts).
`telemetry.trace_sample_interval_seconds` is the higher-cadence sampling
interval used only while one or more requests are active; it supplies observed
per-request peak VRAM and the detailed working-set graph.
`telemetry.log_ticket_ttl_seconds` is how long a short-lived SSE
log-stream ticket (minted via `POST /v1/telemetry/logs-ticket`) stays
valid; the dashboard's `EventSource` client uses one of these instead of
putting the admin token in the URL. See `docs/TELEMETRY.md` for measurement
semantics, CLI commands, and the full endpoint surface.

### federation (coordinator: `muse federate`)

| key | env | default |
|---|---|---|
| `federation.refresh_interval_seconds` | `MUSE_FEDERATION_REFRESH_INTERVAL_SECONDS` | 3.0 |
| `federation.forward_timeout_seconds` | `MUSE_FEDERATION_FORWARD_TIMEOUT_SECONDS` | 300.0 |
| `federation.poll_timeout_seconds` | `MUSE_FEDERATION_POLL_TIMEOUT_SECONDS` | 10.0 |
| `federation.config_file` | `MUSE_FEDERATION_CONFIG` | null |

`federation.refresh_interval_seconds` is how often the coordinator's
background task polls each node's `/v1/models` + `/health` (+ gated
`/v1/telemetry/summary` if a per-node token is configured) and refreshes
its cached routing snapshot. `federation.forward_timeout_seconds` bounds
each request the coordinator forwards to a node; it defaults high
(300s) because generation requests can run for minutes.
`federation.poll_timeout_seconds` bounds each per-node state poll; it
defaults to 10s, deliberately ABOVE a node's own
`server.aggregation_timeout_seconds` (5s), so a node that is briefly slow
to aggregate its workers' `/v1/models` (e.g. a CPU node with a worker
mid-generation) is not falsely marked unreachable and dropped from
routing. `federation.config_file` is an explicit path to the node-list yaml; when
unset, `muse federate` falls back to `<catalog_dir>/federation.yaml` if
that file exists, else nodes come from `--node` CLI entries alone. See
the "Federation" section of `CLAUDE.md` for the coordinator design.

## Adding a new setting

Add one `Setting(...)` row to `SETTINGS` in `src/muse/core/config.py`, then read
it at the call site with `config.get("group.key")`. A meta-test
(`tests/core/test_no_stray_env_reads.py`) fails if any code reads a `MUSE_*` env
var directly outside `muse.core.config`, so new settings must go through the
registry.
