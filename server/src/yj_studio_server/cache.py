from __future__ import annotations

from pathlib import Path


CacheEntry = tuple[float, Path, int]


def plan_slice_cache_removals(entries: list[CacheEntry], budget_bytes: int) -> tuple[list[CacheEntry], int]:
    total = sum(size for _mtime, _path, size in entries)
    planned: list[CacheEntry] = []
    for entry in sorted(entries, key=lambda item: item[0]):
        if total <= budget_bytes:
            break
        planned.append(entry)
        total -= entry[2]
    return planned, max(0, int(total))


def enforce_slice_cache_budget(cache_dir: str | Path, budget_bytes: int) -> dict[str, int]:
    """Trim oldest cached slice files until the directory is under budget."""
    root = Path(cache_dir)
    if budget_bytes <= 0 or not root.exists():
        return {"removed": 0, "remaining_bytes": 0, "budget_bytes": int(budget_bytes)}

    files: list[CacheEntry] = []
    for path in root.glob("*.npy"):
        try:
            stat = path.stat()
        except OSError:
            continue
        size = int(stat.st_size)
        files.append((float(stat.st_mtime), path, size))

    planned, remaining_after_plan = plan_slice_cache_removals(files, budget_bytes)
    removed = 0
    skipped_bytes = 0
    for _mtime, path, size in planned:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            skipped_bytes += size
            continue
        removed += 1

    return {
        "removed": removed,
        "remaining_bytes": int(remaining_after_plan + skipped_bytes),
        "budget_bytes": int(budget_bytes),
    }
