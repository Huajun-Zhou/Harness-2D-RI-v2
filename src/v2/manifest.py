"""
v2.manifest — run manifest logger for Phase 2+ experiments.

Every experiment run produces a JSON manifest that records the exact code,
config, data, and seed state that generated the outputs in that run
directory. Reading a manifest is sufficient to reconstruct which git commit
to check out, which data file to load, and which seed to pass to re-produce
the run bit-for-bit (modulo hardware-level non-determinism inside PySR,
which is acknowledged separately in the paper).

Contents of every manifest:
  - run_id           : human-readable string (e.g. "E2_olo_O_seed42")
  - run_seed         : integer, the root seed
  - experiment       : short name of the experiment (E1, E2, ...)
  - timestamp_start  : ISO-8601 UTC
  - timestamp_end    : ISO-8601 UTC (filled in by `finalise()`)
  - git              : commit hash, short hash, branch, dirty flag
  - python_version   : X.Y.Z
  - pkg_versions     : dict of relevant package versions
  - config_path      : absolute path
  - config_sha256    : hash of the config file contents
  - data             : dict of input data files and their sha256
  - cmdline          : sys.argv
  - env_summary      : a small whitelisted subset of env vars (no secrets)
  - notes            : free-form string

Never put secrets (API keys, tokens, cookies) in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_WHITELIST_ENV = (
    "PYSR_VERSION",
    "JULIA_VERSION",
    # Note: ANTHROPIC_API_KEY is explicitly NOT whitelisted.
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> dict:
    """Best-effort git metadata; returns {} if not a git repo."""
    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                args, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return {"repo": False}
    short = _run(["git", "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "repo": True,
        "commit": commit,
        "short": short,
        "branch": branch,
        "dirty": bool(status),
    }


def _pkg_versions() -> dict:
    out: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "sklearn", "pysr", "anthropic"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", None)
        except ImportError:
            out[name] = None
    return out


def _env_summary() -> dict:
    return {k: os.environ.get(k) for k in _WHITELIST_ENV if os.environ.get(k)}


@dataclass
class RunManifest:
    run_id: str
    experiment: str
    run_seed: int
    config_path: str
    data_files: list[str] = field(default_factory=list)
    notes: str = ""

    # populated by `build()` / `finalise()`
    timestamp_start: str = ""
    timestamp_end: str = ""
    git: dict = field(default_factory=dict)
    python_version: str = ""
    platform: str = ""
    pkg_versions: dict = field(default_factory=dict)
    config_sha256: str = ""
    data: dict = field(default_factory=dict)
    cmdline: list[str] = field(default_factory=list)
    env_summary: dict = field(default_factory=dict)

    def build(self) -> "RunManifest":
        """Fill in all the expensive/external fields. Call once at run start."""
        self.timestamp_start = datetime.now(timezone.utc).isoformat()
        self.git = _git_info()
        self.python_version = platform.python_version()
        self.platform = platform.platform()
        self.pkg_versions = _pkg_versions()
        cfg = Path(self.config_path)
        self.config_sha256 = _sha256_file(cfg) if cfg.is_file() else ""
        self.data = {}
        for df in self.data_files:
            p = Path(df)
            self.data[df] = _sha256_file(p) if p.is_file() else "MISSING"
        self.cmdline = list(sys.argv)
        self.env_summary = _env_summary()
        return self

    def finalise(self) -> "RunManifest":
        """Stamp the end time. Call after the experiment finishes successfully."""
        self.timestamp_end = datetime.now(timezone.utc).isoformat()
        return self

    def write(self, out_dir: str | Path) -> Path:
        """Write `manifest.json` into `out_dir`; returns the written path."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "manifest.json"
        with path.open("w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
        return path
