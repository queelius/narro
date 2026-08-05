# Request telemetry and GPU working-set traces

Muse records structured request, model-lifecycle, and resource events in
`<catalog_dir>/telemetry.db`. The store uses SQLite WAL mode and is safe to
query while `muse serve` is running. Retention defaults to seven days.

The dashboard at `/dashboard` and the `muse telemetry` CLI use the same store
queries. They show:

- end-to-end request latency, with queue, cold-load, and worker-forward spans;
- an exact cold/hot label for each request;
- LRU evictions performed on behalf of that request;
- peak device-wide VRAM observed while the request was active;
- higher-cadence VRAM-over-time samples while requests are active; and
- the live resident model set, queue depth, request counts, and worker logs.

Device-wide VRAM is intentional. The central working-set claim is about all
models resident on the GPU, not memory allocated by one Python worker. On a
CPU-only host or when NVML is unavailable, VRAM values remain null rather than
being reported as zero.

For streaming responses, end-to-end and forward latency run until the stream
closes (normal completion, downstream disconnect, or upstream failure). A
trace's `stream` field still distinguishes streaming and buffered requests.

## CLI

```bash
muse telemetry status
muse telemetry summary --since 24h
muse telemetry traces --since 6h --limit 100
muse telemetry traces --model kokoro-82m --json
muse telemetry vram --since 1h
muse telemetry export trace.json --since 7d --format json
muse telemetry export trace.csv --since 7d --format csv
muse telemetry prune --older-than 14d --dry-run
muse telemetry prune --older-than 14d
```

Exports refuse to overwrite an existing path unless `--force` is supplied.

`summary` produces the cold/hot comparison table suited to a benchmark or blog
post. `traces` retains individual measurements. `vram` renders a simple
terminal graph. `export` is the safest way to move a consistent snapshot off a
running remote node without copying SQLite WAL files directly.

Summary rows sourced from the new request fields are labeled `measured`.
Pre-upgrade rows are labeled `estimated`: Muse combines recorded model-load
duration with historical forward latency and parses lifecycle eviction reasons.
It does not silently present that reconstruction as an exact request trace.

## Dashboard authentication

Telemetry APIs require `admin.token` by default. The static dashboard shell is
always loadable, then authenticates its API calls with that token.

For an explicitly trusted development or loopback-only deployment, the token
requirement can be disabled independently of the admin API:

```yaml
telemetry:
  require_auth: false
```

or:

```bash
MUSE_TELEMETRY_REQUIRE_AUTH=false muse serve --host 127.0.0.1
```

Do not disable telemetry authentication on a publicly reachable bind. Request
metadata, model names, resource history, and worker logs can contain sensitive
operational information. The default remains `true`.

Relevant settings:

```yaml
telemetry:
  enabled: true
  require_auth: true
  retention_days: 7
  sample_interval_seconds: 10.0
  trace_sample_interval_seconds: 0.25
  log_buffer_kb: 64
```

The ordinary interval controls idle/background samples. While at least one
request is active, the sampler switches to `trace_sample_interval_seconds`,
then returns to the lower idle cadence automatically.

## Existing installations and historical recovery

An existing `telemetry.db` is migrated in place when the supervisor or CLI
opens it. Events written before request tracing was added remain queryable, but
their new trace fields are null. They contain useful request counts, model-load
and eviction events, forward latency, and free-VRAM samples. Cold/hot status
can sometimes be approximated by correlating a request with a nearby
`model_load`, but it cannot be made exact retroactively under concurrent load.

Outside Muse, systemd/journald or a retained `muse-serve.log` may recover HTTP
request timestamps/statuses and worker lifecycle messages. Historical GPU
memory is only available if a tool such as `nvidia-smi dmon`, DCGM, Prometheus,
or another GPU monitor was already persisting it. NVIDIA drivers do not expose
a queryable VRAM history after the fact.

For a remote node, prefer running the read-only CLI there and exporting:

```bash
ssh <node> 'muse telemetry status'
ssh <node> 'muse telemetry summary --since 7d'
ssh <node> 'muse telemetry export /tmp/muse-trace.json --since 7d'
```

Then copy the exported file through the deployment's normal secure channel.
