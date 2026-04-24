"""
v2.seeds — deterministic seed derivation for reproducibility.

Every Phase 2+ experiment gets a single `run_seed` (recorded in the manifest).
All downstream seeds (PySR, CV, outer-fold, bootstrap, inner-split) are
computed deterministically from the run_seed via a salted hash. This means:

  1. If you know the run_seed, you can reconstruct every intermediate seed.
  2. Two experiments with different run_seeds give fully independent samples.
  3. Never call `np.random.seed()` or `random.seed()` anywhere else — always
     obtain a sub-seed from `SeedManager`.

Usage::

    sm = SeedManager(run_seed=42)
    pysr_seed = sm.derive("pysr", fold=3)          # outer-LOCO fold 3
    cv_seed   = sm.derive("cv_kfold")
    boot_seed = sm.derive("bootstrap", replicate=17)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


# Keep seeds in 32-bit unsigned range — widest common denominator across
# numpy, sklearn, PyTorch, PySR, Julia.
_SEED_MAX = 2**32 - 1


@dataclass(frozen=True)
class SeedManager:
    """Deterministic sub-seed factory rooted at a single `run_seed`."""

    run_seed: int

    def derive(self, purpose: str, **kwargs: Any) -> int:
        """Return a deterministic sub-seed for `purpose` and any kwargs.

        Example: ``derive("pysr", fold=3, inner_seed=0)`` always returns the
        same int across Python runs and across machines.

        Implementation: SHA-256 of ``{run_seed}|{purpose}|{sorted kwargs}``
        taken modulo 2^32. SHA-256 is overkill for seed derivation but
        guarantees (a) cross-platform determinism and (b) no accidental
        collisions from two different purposes hashing to the same int,
        which a naive ``hash()`` would allow.
        """
        key_parts = [str(self.run_seed), purpose]
        for k in sorted(kwargs):
            key_parts.append(f"{k}={kwargs[k]}")
        key = "|".join(key_parts).encode("utf-8")
        digest = hashlib.sha256(key).digest()
        return int.from_bytes(digest[:4], byteorder="big") % _SEED_MAX


def split_seed(run_seed: int, purpose: str, **kwargs: Any) -> int:
    """Functional shortcut for `SeedManager(run_seed).derive(purpose, **kwargs)`."""
    return SeedManager(run_seed).derive(purpose, **kwargs)
