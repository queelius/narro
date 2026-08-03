# Storage lifecycle

Muse keeps model environments and newly pulled weights below the configured
catalog directory (`~/.muse` by default):

```text
~/.muse/
  catalog.json
  venvs/<model-id>/
  weights/<artifact-or-Hugging-Face-cache>/
```

Older Muse releases put bundled-model weights in the user-wide Hugging Face
cache. Those legacy paths remain usable and are reported, but Muse never
deletes them because other applications may share the same blobs. New pulls
use the Muse-owned weights root so their lifecycle can be managed safely.

## Inspecting storage

```bash
muse doctor storage
muse doctor storage --json
```

The report distinguishes:

- catalog-referenced venvs and weights;
- old incomplete downloads and abandoned staging workspaces;
- unreferenced resources, which may have been intentionally retained by
  `muse models remove`;
- missing or unsafe catalog paths;
- user-wide Hugging Face and pip caches, reported separately;
- filesystem free space.

The doctor is read-only. It exits `1` when it finds actionable storage health
conditions and `2` when state cannot be inspected safely.

## Reclaiming space

```bash
# Delete old incomplete downloads and definitively abandoned staging data.
muse storage prune

# Preview the identical selection without changing files.
muse storage prune --dry-run

# Also consider unregistered Muse-owned venvs and weights.
muse storage prune --include-unreferenced --dry-run
muse storage prune --include-unreferenced

# Change the default 24-hour grace period.
muse storage prune --older-than-hours 168
```

Default pruning is deliberately narrow. It never evicts a cataloged model,
even when disabled, and it never touches shared Hugging Face or pip caches.
`--include-unreferenced` is explicit because current catalogs cannot
distinguish an accidental orphan from files deliberately retained for a fast
future re-pull.

Cleanup and pulls coordinate through process and filesystem locks. Before a
planned path is deleted, Muse checks its filesystem identity, newest descendant
timestamp, and current catalog references again. A venv transaction workspace
is preserved when the canonical venv is missing or incomplete because that
workspace may contain the only recoverable prior environment.

## Automatic maintenance

Before a model pull, Muse checks filesystem headroom. By default, if free space
falls below either 50 GiB or 5%, Muse automatically runs the same narrow safe
cleanup for transient data untouched for at least 24 hours. It never includes
unreferenced model environments or weights, and cleanup problems are reported
without hiding the pull's own result.

The policy is configurable through `config.yaml`, `muse config set`, or the
corresponding environment variables:

```yaml
storage:
  auto_prune_before_pull: true
  auto_prune_grace_hours: 24
  auto_prune_min_free_gb: 50
  auto_prune_min_free_percent: 5
```

Set `storage.auto_prune_before_pull` to `false` to disable it. Explicit
`muse storage prune --include-unreferenced` remains the only bulk cleanup that
can remove retained/unregistered model resources.

## Shared caches

The reported Hugging Face and pip caches are user-wide and may be shared with
other applications, so Muse never deletes from them. After reviewing their
contents, manage them with their owning tools:

```bash
hf cache ls
hf cache prune
python -m pip cache info
python -m pip cache purge
```

`hf cache prune` targets detached revisions and incomplete downloads; use
`hf cache rm` for explicitly selected repositories or revisions. Pip's purge
removes the whole pip download/wheel cache, not installed packages.

## Removing one model

```bash
muse models remove MODEL_ID
muse models remove MODEL_ID --purge
```

Without `--purge`, Muse unregisters the model and retains its files. With
`--purge`, Muse deletes its per-model venv and unshared Muse-owned weights.
Hugging Face snapshots are cleaned at repository scope so the sibling `blobs/`
directory is reclaimed; a repository is preserved while any other catalog
entry references one of its revisions. Legacy shared-cache weights are always
preserved.

## Why venvs are still large

Cleanup removes stale data but does not change the current isolation model:
each model has a dedicated environment, often including its own Torch, Triton,
and CUDA runtime packages. A future environment-store migration can preserve
dependency isolation while deduplicating identical immutable package payloads.
A shared mutable base environment is intentionally avoided because one model's
upgrade could break another model's runtime.
