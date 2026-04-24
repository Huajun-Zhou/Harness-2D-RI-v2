"""
Real Data Loader
================
Loads actual 2D TMD material data from three real sources:

  1. JARVIS-DFT (available immediately via `pip install jarvis-tools`)
     ~1,100 2D monolayers with static dielectric tensors.
     n ≈ sqrt(epsx_el) → derive Cauchy A from static value.

  2. C2DB (DTU) — requires downloading c2db.db from https://c2db.fysik.dtu.dk/
     16,789 materials; RPA polarizability → Clausius-Mossotti → n;
     BSE absorbance spectra for ~300 materials → fit Cauchy A, B.

  3. RefractiveIndex.INFO (experimental)
     Clone https://github.com/polyanskiy/refractiveindex.info-database
     10-15 TMD monolayers with measured n(λ) curves → fit Cauchy A, B.

Usage
-----
  from src.real_data_loader import RealDataLoader

  loader = RealDataLoader()

  # Try JARVIS (no download needed):
  df = loader.load_jarvis()

  # Try C2DB (requires c2db.db on disk):
  df = loader.load_c2db('path/to/c2db.db')

  # Try RefractiveIndex experimental data:
  df = loader.load_refractiveindex('path/to/refractiveindex.info-database')

  # Auto-select best available source:
  df = loader.load_best_available()
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger("HARNESS.RealDataLoader")

# ─────────────────────────────────────────────────────────────────────────────
# Periodic-table reference data (same as data_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────
ELEMENT_DATA = {
    # (Z, electronegativity_Pauling, atomic_radius_Å, IE1_eV, d_electrons)
    'Ti': (22, 1.54, 1.47, 6.83, 2),  'Zr': (40, 1.33, 1.60, 6.63, 2),
    'Hf': (72, 1.30, 1.59, 6.83, 2),  'V':  (23, 1.63, 1.34, 6.75, 3),
    'Nb': (41, 1.60, 1.46, 6.76, 4),  'Ta': (73, 1.50, 1.46, 7.55, 3),
    'Cr': (24, 1.66, 1.28, 6.77, 5),  'Mo': (42, 2.16, 1.39, 7.09, 5),
    'W':  (74, 2.36, 1.39, 7.86, 4),  'Mn': (25, 1.55, 1.27, 7.43, 5),
    'Re': (75, 1.90, 1.37, 7.83, 5),  'Fe': (26, 1.83, 1.26, 7.90, 6),
    'Co': (27, 1.88, 1.25, 7.88, 7),  'Ni': (28, 1.91, 1.24, 7.64, 8),
    'Pd': (46, 2.20, 1.37, 8.34, 10), 'Pt': (78, 2.28, 1.36, 8.96, 9),
    'Sc': (21, 1.36, 1.62, 6.56, 1),
}
CHALCOGEN_DATA = {
    # (Z, electronegativity_Pauling, atomic_radius_Å, EA_eV)
    'S':  (16, 2.58, 1.05, 2.08),
    'Se': (34, 2.55, 1.20, 2.02),
    'Te': (52, 2.10, 1.38, 1.97),
}

# TMD compositions: metal → chalcogen symbol embedded in formula strings
_TMD_METALS    = set(ELEMENT_DATA.keys())
_TMD_CHALCOGENS = set(CHALCOGEN_DATA.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Helper: fit Cauchy coefficients from n(λ) tabular data
# ─────────────────────────────────────────────────────────────────────────────
def fit_cauchy(wavelengths_nm: np.ndarray, n_values: np.ndarray
               ) -> Tuple[float, float]:
    """
    Fit Cauchy model  n(λ) = A + B / λ²  (λ in nm, B in nm²).

    Returns (A, B).  Uses least-squares linear regression.
    """
    lam2_inv = 1.0 / (wavelengths_nm ** 2)   # 1/λ²
    # Linear system: [1, 1/λ²] @ [A, B]ᵀ = n
    X = np.column_stack([np.ones(len(wavelengths_nm)), lam2_inv])
    coeffs, *_ = np.linalg.lstsq(X, n_values, rcond=None)
    A, B = float(coeffs[0]), float(coeffs[1])
    return A, B


def _parse_tmd_formula(formula: str) -> Optional[Tuple[str, str]]:
    """
    Try to parse a chemical formula and identify the TMD metal and chalcogen.
    Returns (metal, chalcogen) or None.

    Handles: 'MoS2', 'WS2', 'MoSe2', 'WSe2', 'MoTe2', 'TiS2', etc.
    """
    import re
    for metal in sorted(_TMD_METALS, key=len, reverse=True):   # longest first
        if formula.startswith(metal):
            rest = formula[len(metal):]
            for chalc in _TMD_CHALCOGENS:
                if rest.startswith(chalc):
                    return metal, chalc
    return None


def _element_row(metal: str, chalcogen: str) -> Optional[Dict]:
    """Build the standard feature dict for a TMD from element tables."""
    if metal not in ELEMENT_DATA or chalcogen not in CHALCOGEN_DATA:
        return None
    Z_M, chi_M, r_M, IE_M, d_e = ELEMENT_DATA[metal]
    Z_X, chi_X, r_X, EA_X = CHALCOGEN_DATA[chalcogen]
    return dict(
        metal=metal, chalcogen=chalcogen,
        atomic_number_M=Z_M, atomic_number_X=Z_X,
        electronegativity_M=chi_M, electronegativity_X=chi_X,
        atomic_radius_M=r_M, atomic_radius_X=r_X,
        ionization_energy_M=IE_M, electron_affinity_X=EA_X,
        d_electron_count=d_e,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main loader class
# ─────────────────────────────────────────────────────────────────────────────
class RealDataLoader:
    """
    Unified loader for real 2D TMD datasets.

    Each `load_*` method returns a DataFrame compatible with
    TMDDataPipeline's downstream feature engineering (_compute_derived_features).

    Mandatory columns every method must produce:
        metal, chalcogen,
        atomic_number_M, atomic_number_X,
        electronegativity_M, electronegativity_X,
        atomic_radius_M, atomic_radius_X,
        ionization_energy_M, electron_affinity_X,
        d_electron_count,
        bond_length_MX, lattice_constant_a, monolayer_thickness,
        coordination_number, point_group_encoded,
        cauchy_A,              ← primary target
        cauchy_B,              ← secondary target
        n_at_1064nm,           ← derived from Cauchy fit
        n_at_785nm,
        data_source            ← provenance tag
    """

    # ------------------------------------------------------------------ #
    # 1. JARVIS-DFT                                                        #
    # ------------------------------------------------------------------ #
    def load_jarvis(self, min_samples: int = 20) -> pd.DataFrame:
        """
        Load 2D TMD data from JARVIS-DFT via jarvis-tools.

        Requires:  pip install jarvis-tools
        No signup or download needed — data fetched from Figshare automatically
        and cached locally under ~/.jarvis/.

        Strategy for optical target:
          JARVIS provides the *static* dielectric tensor (epsx, epsy, epsz).
          For visible/NIR TMDs the electronic part dominates, so:
              n_static ≈ sqrt(epsx_el)
          We then fit a simplified Cauchy model by using the known
          dispersion relationship for TMDs to estimate B from the bandgap:
              B ≈ A * λ_gap² / (1 - (λ/λ_gap)²)   (single-oscillator)
          where λ_gap = hc / E_gap (nm).
          This gives physically reasonable Cauchy A and B.
        """
        try:
            from jarvis.db.figshare import data as jdata
        except ImportError:
            raise ImportError(
                "jarvis-tools is not installed.\n"
                "Run:  pip install jarvis-tools"
            )

        logger.info("Downloading JARVIS-DFT 2D dataset (cached after first run)…")
        raw = jdata(dataset='dft_2d')
        logger.info(f"  JARVIS dft_2d: {len(raw)} total entries")

        records = []
        skipped = 0
        for entry in raw:
            formula = entry.get('formula', '')
            parsed = _parse_tmd_formula(formula)
            if parsed is None:
                skipped += 1
                continue
            metal, chalcogen = parsed

            row = _element_row(metal, chalcogen)
            if row is None:
                skipped += 1
                continue

            # Static dielectric → n_static
            # JARVIS uses 'na' string for missing values — filter these out
            def _safe_float(v):
                try:
                    f = float(v)
                    return f if np.isfinite(f) else None
                except (TypeError, ValueError):
                    return None

            epsx = _safe_float(entry.get('epsx'))
            mepsx = _safe_float(entry.get('mepsx'))
            eps_el = mepsx if (mepsx is not None and mepsx > 1.0) else epsx
            if eps_el is None or eps_el <= 1.0:
                skipped += 1
                continue
            n_static = float(np.sqrt(eps_el))

            # Bandgap for single-oscillator dispersion estimate
            eg = _safe_float(entry.get('optb88vdw_bandgap')) \
                 or _safe_float(entry.get('mbj_bandgap'))
            if eg and eg > 0.1:
                eg_eV = eg
                lam_gap_nm = 1239.8 / eg_eV          # E → wavelength
                # Single-oscillator Sellmeier → Cauchy approximation
                # For λ << λ_gap: n ≈ n_0 (1 + λ²/(2*(λ_gap²-λ²)))
                # At λ = 800 nm (mid-NIR window):
                lam_ref = 800.0
                if lam_gap_nm > lam_ref * 1.2:       # only valid below gap
                    denom = lam_gap_nm**2 - lam_ref**2
                    B_est = n_static * lam_ref**2 * lam_gap_nm**2 / (2 * denom * lam_gap_nm**2)
                    B_est = max(B_est, 0.0)
                else:
                    B_est = 0.0
                cauchy_A = n_static - B_est / lam_ref**2
                cauchy_B = B_est
            else:
                cauchy_A = n_static        # static value as Cauchy A
                cauchy_B = 0.0

            # Clip to physical range
            if not (1.3 <= cauchy_A <= 6.0):
                skipped += 1
                continue

            # Structural data from JARVIS atoms object
            bond_length, lattice_a, thickness, coord_n = (
                self._extract_structure_jarvis(entry)
            )
            row.update(dict(
                bond_length_MX=bond_length,
                lattice_constant_a=lattice_a,
                monolayer_thickness=thickness,
                coordination_number=coord_n,
                point_group_encoded=self._encode_point_group_jarvis(entry),
                cauchy_A=round(cauchy_A, 6),
                cauchy_B=round(cauchy_B, 6),
                n_at_1064nm=round(cauchy_A + cauchy_B / 1064**2, 6),
                n_at_785nm=round(cauchy_A + cauchy_B / 785**2, 6),
                jid=entry.get('jid', ''),
                data_source='JARVIS-DFT',
            ))
            records.append(row)

        df = pd.DataFrame(records)
        logger.info(
            f"  JARVIS TMDs extracted: {len(df)} records "
            f"({skipped} skipped — non-TMD or missing optical data)"
        )
        if len(df) < min_samples:
            logger.warning(
                f"  Only {len(df)} samples — below min_samples={min_samples}"
            )
        return df

    def _extract_structure_jarvis(self, entry: dict
                                   ) -> Tuple[float, float, float, int]:
        """Extract bond_length, lattice_a, thickness, coord_number from JARVIS entry."""
        try:
            from jarvis.core.atoms import Atoms as JAtoms
            atoms = JAtoms.from_dict(entry['atoms'])
            lat = atoms.lattice_mat
            a = float(np.linalg.norm(lat[0]))      # in-plane lattice constant a

            # Estimate M-X bond length from positions
            coords = np.array(atoms.cart_coords)
            elements = atoms.elements
            m_idx = [i for i, e in enumerate(elements) if e in _TMD_METALS]
            x_idx = [i for i, e in enumerate(elements) if e in _TMD_CHALCOGENS]
            if m_idx and x_idx:
                dists = [np.linalg.norm(coords[m] - coords[x])
                         for m in m_idx for x in x_idx]
                bond_len = float(np.min(dists))
            else:
                bond_len = 2.4   # fallback

            # Monolayer thickness: z-range of chalcogen atoms
            if x_idx:
                z_vals = coords[x_idx, 2]
                thickness = float(np.max(z_vals) - np.min(z_vals)) + 0.5
            else:
                thickness = 3.1  # fallback

            coord_n = 6   # typical for TMDs
            return round(bond_len, 4), round(a, 4), round(thickness, 4), coord_n

        except Exception:
            return 2.4, 3.2, 3.1, 6

    def _encode_point_group_jarvis(self, entry: dict) -> int:
        """Encode point group as integer (1-4 for common TMD symmetries)."""
        sg = entry.get('spg_number', 0)
        if sg in (187, 189):   return 3   # P-6m2 (H-phase)
        if sg in (164,):       return 2   # P-3m1 (T-phase)
        if sg in (11, 14):     return 1   # monoclinic
        return 4

    # ------------------------------------------------------------------ #
    # 2. C2DB (ASE database file)                                          #
    # ------------------------------------------------------------------ #
    def load_c2db(self, db_path: str) -> pd.DataFrame:
        """
        Load from a local C2DB ASE database file (c2db.db).

        Download from: https://c2db.fysik.dtu.dk/  (requires registration)
        or try the DTU Figshare archive:
          https://data.dtu.dk/articles/dataset/Computational_2D_Materials_Database_C2DB_/14616660

        This method extracts:
          - RPA static polarizability → Clausius-Mossotti → n
          - BSE absorbance spectra (where available) → fit Cauchy A, B
          - Structural data (lattice constant, bond lengths, thickness)
        """
        try:
            from ase.db import connect as ase_connect
        except ImportError:
            raise ImportError(
                "ASE is not installed.\n"
                "Run:  pip install ase"
            )

        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(
                f"C2DB database not found at {db_path}\n"
                "Download from: https://c2db.fysik.dtu.dk/\n"
                "or:  https://data.dtu.dk/articles/dataset/"
                "Computational_2D_Materials_Database_C2DB_/14616660"
            )

        logger.info(f"Loading C2DB from {db_path} …")
        db = ase_connect(str(db_path))

        records = []
        skipped = 0

        for row in db.select():
            formula = row.get('formula', row.get('formula_unit', ''))
            parsed = _parse_tmd_formula(str(formula))
            if parsed is None:
                skipped += 1
                continue
            metal, chalcogen = parsed

            base = _element_row(metal, chalcogen)
            if base is None:
                skipped += 1
                continue

            # 1. Try BSE/RPA frequency-dependent data first
            cauchy_A, cauchy_B = self._c2db_bse_cauchy(row)

            # 2. Fall back to static polarizability via Clausius-Mossotti
            if cauchy_A is None:
                cauchy_A, cauchy_B = self._c2db_static_polarizability(row)

            if cauchy_A is None or not (1.3 <= cauchy_A <= 6.5):
                skipped += 1
                continue

            # Structural data
            try:
                atoms = row.toatoms()
                bond_len, lat_a, thickness, coord_n = (
                    self._extract_structure_ase(atoms, metal, chalcogen)
                )
            except Exception:
                bond_len, lat_a, thickness, coord_n = 2.4, 3.2, 3.1, 6

            # Point group
            pg = self._encode_point_group_c2db(row)

            base.update(dict(
                bond_length_MX=round(bond_len, 4),
                lattice_constant_a=round(lat_a, 4),
                monolayer_thickness=round(thickness, 4),
                coordination_number=coord_n,
                point_group_encoded=pg,
                cauchy_A=round(cauchy_A, 6),
                cauchy_B=round(cauchy_B, 6),
                n_at_1064nm=round(cauchy_A + cauchy_B / 1064**2, 6),
                n_at_785nm=round(cauchy_A + cauchy_B / 785**2, 6),
                uid=getattr(row, 'uid', ''),
                data_source='C2DB',
            ))
            records.append(base)

        df = pd.DataFrame(records)
        logger.info(
            f"  C2DB TMDs extracted: {len(df)} records "
            f"({skipped} skipped)"
        )
        return df

    def _c2db_bse_cauchy(self, row) -> Tuple[Optional[float], float]:
        """Try to get Cauchy A, B from BSE absorbance stored in C2DB row data."""
        try:
            import json
            data = getattr(row, 'data', {}) or {}
            # BSE results may be stored as JSON sidecar
            bse_key = 'results-asr.bse.json'
            if bse_key not in data:
                return None, 0.0
            bse = data[bse_key] if isinstance(data[bse_key], dict) \
                else json.loads(data[bse_key])
            # n(ω) array: look for 'n' or 'nw' key
            n_arr = bse.get('n') or bse.get('nw')
            freq = bse.get('frequencies') or bse.get('freqs')   # eV
            if n_arr is None or freq is None:
                return None, 0.0
            n_arr = np.array(n_arr)
            freq = np.array(freq)         # eV
            # Convert to wavelength (nm): λ = 1239.8 / E_eV
            with np.errstate(divide='ignore'):
                lam = 1239.8 / freq
            # Select transparent window: 700–1100 nm, n finite and > 1
            mask = (lam > 700) & (lam < 1100) & np.isfinite(n_arr) & (n_arr > 1.0)
            if np.sum(mask) < 3:
                return None, 0.0
            A, B = fit_cauchy(lam[mask], n_arr[mask])
            if 1.3 <= A <= 6.5 and B >= 0:
                return A, B
        except Exception:
            pass
        return None, 0.0

    def _c2db_static_polarizability(self, row) -> Tuple[Optional[float], float]:
        """
        Derive n from static RPA polarizability via Clausius-Mossotti.

        For a 2D material in the 3D Clausius-Mossotti relation the effective
        medium layer thickness L must be used:
            ε_eff = 1 + α_2D / (ε_0 * L)     (in SI)
        In practice we use: n ≈ sqrt(1 + alpha_x / (volume_per_formula * ε_0))
        where volume ≈ area * L, L ≈ monolayer_thickness + vac_gap/2.
        """
        try:
            alpha_x = (getattr(row, 'alphax_el', None)
                       or getattr(row, 'alpha_x', None))
            if alpha_x is None:
                return None, 0.0
            alpha_x = float(alpha_x)   # Å³ per formula unit
            if alpha_x <= 0:
                return None, 0.0

            # Get area per formula unit from lattice
            atoms = row.toatoms()
            cell = atoms.get_cell()
            area = float(np.linalg.norm(np.cross(cell[0], cell[1])))  # Å²
            n_fu = len([e for e in atoms.get_chemical_symbols()
                        if e in _TMD_METALS])   # formula units per cell
            if n_fu == 0:
                n_fu = 1
            alpha_per_area = alpha_x / (area / n_fu)   # Å (dimensionless scaled)
            # ε_eff via simple polarisability → refractive index
            eps_eff = 1.0 + 4.0 * np.pi * alpha_per_area / 3.0
            n_static = float(np.sqrt(max(eps_eff, 1.0)))
            if not 1.3 <= n_static <= 6.5:
                return None, 0.0
            return n_static, 0.0
        except Exception:
            return None, 0.0

    def _extract_structure_ase(self, atoms, metal: str, chalcogen: str
                                ) -> Tuple[float, float, float, int]:
        """Extract structural parameters from ASE Atoms object."""
        cell = atoms.get_cell()
        a = float(np.linalg.norm(cell[0]))   # in-plane a

        symbols = atoms.get_chemical_symbols()
        coords = atoms.get_positions()

        m_idx = [i for i, s in enumerate(symbols) if s == metal]
        x_idx = [i for i, s in enumerate(symbols) if s == chalcogen]

        if m_idx and x_idx:
            dists = [np.linalg.norm(coords[m] - coords[x])
                     for m in m_idx for x in x_idx]
            bond_len = float(np.min(dists))
        else:
            bond_len = 2.4

        if x_idx:
            z_vals = coords[x_idx, 2]
            thickness = float(np.max(z_vals) - np.min(z_vals)) + 0.5
        else:
            thickness = 3.1

        return round(bond_len, 4), round(a, 4), round(thickness, 4), 6

    def _encode_point_group_c2db(self, row) -> int:
        pg_str = getattr(row, 'spglib_pointgroup', '') or ''
        if '-6m2' in pg_str or '6/mmm' in pg_str:  return 3
        if '-3m' in pg_str:                          return 2
        if 'mmm' in pg_str:                          return 4
        return 1

    # ------------------------------------------------------------------ #
    # 3. RefractiveIndex.INFO (experimental)                              #
    # ------------------------------------------------------------------ #
    def load_refractiveindex(self, db_dir: str) -> pd.DataFrame:
        """
        Load experimental n(λ) data from a local clone of the
        refractiveindex.info-database repository.

        Clone the database:
          git clone https://github.com/polyanskiy/refractiveindex.info-database

        The loader scans the database/data/main/ directory for any material
        whose shelf path matches a known TMD formula, parses the YAML files,
        fits Cauchy A and B, and builds the feature DataFrame.
        """
        db_dir = Path(db_dir)
        data_dir = db_dir / 'database' / 'data' / 'main'
        if not data_dir.exists():
            # Also try direct path if user pointed to database/data/main
            data_dir = db_dir
        if not data_dir.exists():
            raise FileNotFoundError(
                f"RefractiveIndex database not found at {db_dir}\n"
                "Clone with:\n"
                "  git clone "
                "https://github.com/polyanskiy/refractiveindex.info-database"
            )
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")

        # TMD formula → book name mapping in refractiveindex.info
        tmd_books = {
            'MoS2': ('Mo', 'S'), 'MoSe2': ('Mo', 'Se'),
            'MoTe2': ('Mo', 'Te'), 'WS2': ('W', 'S'),
            'WSe2': ('W', 'Se'), 'WTe2': ('W', 'Te'),
            'TiS2': ('Ti', 'S'), 'TiSe2': ('Ti', 'Se'),
            'NbS2': ('Nb', 'S'), 'NbSe2': ('Nb', 'Se'),
            'VS2': ('V', 'S'), 'VSe2': ('V', 'Se'),
        }

        records = []
        for formula, (metal, chalcogen) in tmd_books.items():
            formula_dir = data_dir / formula
            if not formula_dir.exists():
                continue
            for yaml_file in formula_dir.glob('*.yml'):
                try:
                    rec = self._parse_ri_yaml(
                        yaml_file, formula, metal, chalcogen
                    )
                    if rec is not None:
                        records.append(rec)
                except Exception as e:
                    logger.debug(f"  Skip {yaml_file.name}: {e}")

        df = pd.DataFrame(records)
        logger.info(
            f"  RefractiveIndex.INFO: {len(df)} TMD entries loaded"
        )
        return df

    def _parse_ri_yaml(self, yaml_file: Path,
                        formula: str, metal: str, chalcogen: str
                        ) -> Optional[dict]:
        """Parse a single YAML file from refractiveindex.info and build a row."""
        import yaml
        with open(yaml_file) as f:
            doc = yaml.safe_load(f)

        base = _element_row(metal, chalcogen)
        if base is None:
            return None

        # Find tabulated n data
        data_arr = None
        for item in doc.get('DATA', []):
            dtype = item.get('type', '')
            if 'tabulated n' in dtype or 'tabulated nk' in dtype:
                raw = item.get('data', '').strip()
                rows = [line.split() for line in raw.splitlines() if line.strip()]
                arr = np.array([[float(v) for v in r[:2]] for r in rows if len(r) >= 2])
                if len(arr) >= 4:
                    data_arr = arr
                    break

        if data_arr is None:
            return None

        # Column 0: wavelength (µm), column 1: n
        lam_um = data_arr[:, 0]
        n_vals = data_arr[:, 1]
        lam_nm = lam_um * 1000.0

        # Select transparent window 700–1200 nm
        mask = (lam_nm > 700) & (lam_nm < 1200) & np.isfinite(n_vals) & (n_vals > 1.0)
        if np.sum(mask) < 3:
            return None

        A, B = fit_cauchy(lam_nm[mask], n_vals[mask])
        if not (1.3 <= A <= 6.5):
            return None

        # Structural defaults from ELEMENT_DATA (no direct structure in RI.info)
        Z_M, chi_M, r_M, IE_M, d_e = ELEMENT_DATA.get(metal, (0,)*5)
        Z_X, chi_X, r_X, EA_X = CHALCOGEN_DATA.get(chalcogen, (0,)*4)
        bond_len = r_M + r_X
        lat_a = bond_len * 1.75
        thickness = 2 * r_X + 0.5

        base.update(dict(
            bond_length_MX=round(bond_len, 4),
            lattice_constant_a=round(lat_a, 4),
            monolayer_thickness=round(thickness, 4),
            coordination_number=6,
            point_group_encoded=3,   # most common TMD symmetry
            cauchy_A=round(A, 6),
            cauchy_B=round(B, 6),
            n_at_1064nm=round(A + B / 1064**2, 6),
            n_at_785nm=round(A + B / 785**2, 6),
            ri_file=yaml_file.stem,
            data_source='RefractiveIndex.INFO',
        ))
        return base

    # ------------------------------------------------------------------ #
    # 4. Auto-select best available source                                 #
    # ------------------------------------------------------------------ #
    def load_csv(self, csv_path: str, min_samples: int = 20) -> pd.DataFrame:
        """
        Load a pre-featurised CSV produced by scripts/build_extended_dataset.py.

        Expected columns (Tier 2 extended dataset schema):
          Z_M, Z_X, chi_M, chi_X, r_M, r_X, IE_M, EA_X, d_e,
          d_MX, a_lat, t, bandgap_optb88, cauchy_A, ...

        This method renames columns to match the legacy pipeline schema
        (atomic_number_M, electronegativity_M, ...) so downstream components
        (data_pipeline._compute_derived_features, feature selection, agents)
        work without modification.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        raw = pd.read_csv(path)
        logger.info(f"[CSV loader] Read {len(raw)} rows from {csv_path}")

        # Column rename map (extended → legacy)
        rename = {
            'Z_M':  'atomic_number_M',
            'Z_X':  'atomic_number_X',
            'chi_M': 'electronegativity_M',
            'chi_X': 'electronegativity_X',
            'r_M':  'atomic_radius_M',
            'r_X':  'atomic_radius_X',
            'IE_M': 'ionization_energy_M',
            'EA_X': 'electron_affinity_X',
            'd_e':  'd_electron_count',
            'd_MX': 'bond_length_MX',
            'a_lat': 'lattice_constant_a',
            't':    'monolayer_thickness',
            'anion': 'chalcogen',   # legacy pipeline calls the anion "chalcogen"
        }
        df = raw.rename(columns=rename).copy()

        # Synthesise the few columns the legacy pipeline expects but Tier 2
        # doesn't provide. These are only used for metadata / dummy fields.
        if 'coordination_number' not in df.columns:
            # Coordination inferred from stoichiometry: MX2 → 6, MX3 → 6, MX → 4
            df['coordination_number'] = df.get('stoich_ratio', pd.Series([6]*len(df))).apply(
                lambda r: 6 if r >= 2 else 4)
        if 'point_group_encoded' not in df.columns:
            sg_map = {164: 2, 187: 3, 189: 3, 11: 1, 14: 1}
            df['point_group_encoded'] = df.get('spg_number', 0).map(
                lambda sg: sg_map.get(int(sg) if pd.notna(sg) else 0, 4))
        if 'phase' not in df.columns:
            df['phase'] = 0
        if 'cauchy_B' not in df.columns:
            # Use bandgap-derived Cauchy B estimate (single-oscillator)
            if 'bandgap_optb88' in df.columns:
                lam_gap = 1239.8 / df['bandgap_optb88'].clip(lower=0.5)
                lam_ref = 800.0
                denom = (lam_gap**2 - lam_ref**2).clip(lower=1.0)
                B = df['cauchy_A'] * lam_ref**2 / (2 * denom)
                df['cauchy_B'] = B.clip(lower=0.0).round(6)
            else:
                df['cauchy_B'] = 0.0
        if 'n_at_1064nm' not in df.columns:
            df['n_at_1064nm'] = (df['cauchy_A'] + df['cauchy_B'] / 1064**2).round(6)
        if 'n_at_785nm' not in df.columns:
            df['n_at_785nm'] = (df['cauchy_A'] + df['cauchy_B'] / 785**2).round(6)
        if 'data_source' not in df.columns:
            df['data_source'] = 'JARVIS-DFT-Extended'

        required = [
            'atomic_number_M', 'atomic_number_X',
            'electronegativity_M', 'electronegativity_X',
            'atomic_radius_M', 'atomic_radius_X',
            'ionization_energy_M', 'electron_affinity_X',
            'd_electron_count', 'bond_length_MX', 'lattice_constant_a',
            'monolayer_thickness', 'coordination_number', 'point_group_encoded',
            'cauchy_A',
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        # Drop rows with NaN in any required column (keeps only clean samples)
        before = len(df)
        df = df.dropna(subset=required).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.info(f"[CSV loader] Dropped {dropped} rows with NaN in required columns")

        logger.info(
            f"[CSV loader] Final: {len(df)} samples, cauchy_A ∈ "
            f"[{df['cauchy_A'].min():.3f}, {df['cauchy_A'].max():.3f}]"
        )
        if len(df) < min_samples:
            logger.warning(f"Only {len(df)} samples < min_samples={min_samples}")
        return df

    def load_best_available(self,
                             c2db_path: Optional[str] = None,
                             ri_db_path: Optional[str] = None,
                             min_samples: int = 30) -> Tuple[pd.DataFrame, str]:
        """
        Try each real data source in order of richness. Return the first
        that yields at least min_samples rows.

        Priority: C2DB > JARVIS > RefractiveIndex.INFO

        Returns (df, source_name).
        """
        # 1. C2DB (richest)
        if c2db_path:
            try:
                df = self.load_c2db(c2db_path)
                if len(df) >= min_samples:
                    logger.info(f"Using C2DB: {len(df)} samples")
                    return df, 'c2db'
            except Exception as e:
                logger.warning(f"C2DB load failed: {e}")

        # 2. JARVIS (zero-setup)
        try:
            df = self.load_jarvis(min_samples=min_samples)
            if len(df) >= min_samples:
                logger.info(f"Using JARVIS-DFT: {len(df)} samples")
                return df, 'jarvis'
        except Exception as e:
            logger.warning(f"JARVIS load failed: {e}")

        # 3. RefractiveIndex.INFO (experimental, small)
        if ri_db_path:
            try:
                df = self.load_refractiveindex(ri_db_path)
                if len(df) >= 5:
                    logger.info(f"Using RefractiveIndex.INFO: {len(df)} samples")
                    return df, 'refractiveindex'
            except Exception as e:
                logger.warning(f"RefractiveIndex load failed: {e}")

        raise RuntimeError(
            "No real dataset available with sufficient samples.\n\n"
            "Quick start:\n"
            "  pip install jarvis-tools\n"
            "  python main.py --source jarvis --target cauchy_A\n\n"
            "Or provide a C2DB database file:\n"
            "  python main.py --source c2db --c2db-path /path/to/c2db.db"
        )

    # ------------------------------------------------------------------ #
    # 5. Dataset summary                                                   #
    # ------------------------------------------------------------------ #
    def summary(self, df: pd.DataFrame) -> str:
        """Print a summary of the loaded dataset."""
        lines = [
            f"Dataset: {len(df)} samples, source={df['data_source'].iloc[0]}",
            f"  Metals:     {sorted(df['metal'].unique())}",
            f"  Chalcogens: {sorted(df['chalcogen'].unique())}",
            f"  Cauchy A:   [{df['cauchy_A'].min():.3f}, "
            f"{df['cauchy_A'].max():.3f}]  mean={df['cauchy_A'].mean():.3f}",
            f"  Cauchy B:   [{df['cauchy_B'].min():.3f}, "
            f"{df['cauchy_B'].max():.3f}]",
            f"  n at 1064nm:[{df['n_at_1064nm'].min():.3f}, "
            f"{df['n_at_1064nm'].max():.3f}]",
        ]
        return '\n'.join(lines)
