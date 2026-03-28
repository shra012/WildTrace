from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_ndjson(path: Path, records: Iterable[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def append_ndjson(path: Path, record: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def read_ndjson(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".ndjson", ".jsonl"}:
        return read_ndjson(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported record catalog format: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample_id(*parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def fetch_to_path(source_url: str, destination: Path, timeout_seconds: int, user_agent: str) -> None:
    ensure_parent(destination)
    parsed = urlparse(source_url)
    if parsed.scheme == "file":
        source_path = Path(parsed.path)
        shutil.copy2(source_path, destination)
        return
    if parsed.scheme in {"http", "https"}:
        request = Request(source_url, headers={"User-Agent": user_agent})
        with urlopen(request, timeout=timeout_seconds) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        return
    if parsed.scheme == "":
        shutil.copy2(Path(source_url), destination)
        return
    raise ValueError(f"Unsupported source URL scheme for {source_url}")
