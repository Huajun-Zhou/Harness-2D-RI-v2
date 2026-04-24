"""
TMD Data Pipeline
=================
Layer 1 of the Harness: data collection, feature engineering, and preparation.
Handles both real C2DB data and synthetic demo data for development.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger("HARNESS.DataPipeline")

# Periodic table data for TMD feature engineering
ELEMENT_DATA = {
    # Transition Metals: (Z, electronegativity, atomic_radius_Å, IE_eV, d_electrons)
    'Ti': (22, 1.54, 1.47, 6.83, 2),  'Zr': (40, 1.33, 1.60, 6.63, 2),
    'Hf': (72, 1.30, 1.59, 6.83, 2),  'V':  (23, 1.63, 1.34, 6.75, 3),
    'Nb': (41, 1.60, 1.46, 6.76, 4),  'Ta': (73, 1.50, 1.46, 7.55, 3),
    'Cr': (24, 1.66, 1.28, 6.77, 5),  'Mo': (42, 2.16, 1.39, 7.09, 5),
    'W':  (74, 2.36, 1.39, 7.86, 4),  'Mn': (25, 1.55, 1.27, 7.43, 5),
    'Re': (75, 1.90, 1.37, 7.83, 5),  'Fe': (26, 1.83, 1.26, 7.90, 6),
    'Co': (27, 1.88, 1.25, 7.88, 7),  'Ni': (28, 1.91, 1.24, 7.64, 8),
    'Pd': (46, 2.20, 1.37, 8.34, 10), 'Pt': (78, 2.28, 1.36, 8.96, 9),
    'Sc': (21, 1.36, 1.62, 6.56, 1),  'Y':  (39, 1.22, 1.80, 6.22, 1),
    'Rh': (45, 2.28, 1.34, 7.46, 8),  'Ir': (77, 2.20, 1.36, 8.97, 7),
    'Bi': (83, 2.02, 1.48, 7.29, 0),
}

CHALCOGEN_DATA = {
    # Chalcogens: (Z, electronegativity, atomic_radius_Å, EA_eV)
    'S':  (16, 2.58, 1.05, 2.08),
    'Se': (34, 2.55, 1.20, 2.02),
    'Te': (52, 2.10, 1.38, 1.97),
}


class TMDDataPipeline:
    """
    Data pipeline for 2D TMD materials.

    Supports three data sources:
      'synthetic' / demo_mode=True  — physics-based synthetic data (default)
      'jarvis'                       — JARVIS-DFT via jarvis-tools (pip install jarvis-tools)
      'c2db'                         — C2DB ASE database file (requires download)
      'refractiveindex'              — experimental n(λ) YAML clone
    """

    def __init__(self, data_config: dict, demo_mode: bool = False,
                 data_source: str = 'synthetic',
                 c2db_path: Optional[str] = None,
                 ri_db_path: Optional[str] = None,
                 csv_path: Optional[str] = None):
        self.config = data_config
        self.demo_mode = demo_mode
        # data_source overrides demo_mode if explicitly set to a real source
        self.data_source = data_source if not demo_mode else 'synthetic'
        self.c2db_path = c2db_path
        self.ri_db_path = ri_db_path
        self.csv_path = csv_path
        self.df = None

    def prepare(self, target: str) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for symbolic regression.

        Returns:
            X_train, y_train, X_test, y_test, feature_names
        """
        if self.data_source == 'synthetic' or self.demo_mode:
            self.df = self._generate_synthetic_tmd_data()
        else:
            self.df = self._load_real_data()

        # Feature engineering
        self.df = self._compute_derived_features(self.df)

        # Select features
        feature_cols = self._select_features()
        X = self.df[feature_cols].values
        y = self.df[target].values

        # Train-test split (80/20, stratified by metal type if possible)
        n = len(X)
        np.random.seed(42)
        idx = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, test_idx = idx[:split], idx[split:]

        logger.info(f"Dataset shape: {X.shape}, Target: {target}")
        logger.info(f"Target range: [{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")

        return (X[train_idx], y[train_idx],
                X[test_idx], y[test_idx],
                feature_cols)

    def _generate_synthetic_tmd_data(self) -> pd.DataFrame:
        """
        Generate synthetic TMD data with realistic structure-RI correlations.
        
        Based on physical insights from the ACS Nano paper:
        - Higher atomic number metals → higher polarizability → higher n
        - Heavier chalcogens (Te > Se > S) → higher n
        - Smaller bandgap materials → higher n in near-IR
        """
        logger.info("Generating synthetic TMD dataset (demo mode)...")
        np.random.seed(42)

        records = []
        metals = ['Mo', 'W', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta',
                  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Pd', 'Pt', 'Re', 'Sc']
        chalcogens = ['S', 'Se', 'Te']

        for M in metals:
            for X in chalcogens:
                # Multiple phases/polytypes per composition
                for phase_idx in range(np.random.randint(1, 4)):
                    Z_M, chi_M, r_M, IE_M, d_e = ELEMENT_DATA[M]
                    Z_X, chi_X, r_X, EA_X = CHALCOGEN_DATA[X]

                    # Realistic bond length estimation
                    bond_length = r_M + r_X + np.random.normal(0, 0.03)

                    # Lattice constant (empirical scaling)
                    a = bond_length * (1.7 + 0.1 * np.random.randn())

                    # Monolayer thickness
                    thickness = 2 * r_X + 0.5 + np.random.normal(0, 0.1)

                    # Coordination number (mostly 6 for TMDs, sometimes 3)
                    coord = np.random.choice([3, 6], p=[0.2, 0.8])

                    # Point group encoding
                    pg = np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.2, 0.2])

                    # --- Physics-based RI model ---
                    # Key insight: n correlates with polarizability, which scales with
                    # atomic number, bond polarity, and d-electron count
                    chi_diff = abs(chi_M - chi_X)
                    polarity = (chi_X - chi_M) / (chi_X + chi_M)

                    # Cauchy A: baseline refractive index (dominant term)
                    # Physical model: A ~ f(Z_X, d_electrons, chi_diff, bond_length)
                    cauchy_A = (1.5
                               + 0.025 * Z_X          # Heavier chalcogen → higher n
                               + 0.015 * d_e           # More d-electrons → higher polarizability
                               + 0.8 * polarity        # Bond polarity contributes
                               - 0.1 * bond_length     # Shorter bonds → denser packing
                               + 0.003 * Z_M           # Metal atomic number
                               + np.random.normal(0, 0.08))

                    # Cauchy B: dispersion strength (×10⁴ nm²)
                    cauchy_B = (0.5
                               + 0.02 * Z_X
                               + 0.01 * d_e
                               + 0.3 * chi_diff
                               + np.random.normal(0, 0.05))

                    # Refractive indices at specific wavelengths
                    n_1064 = cauchy_A + cauchy_B / (1064**2) * 1e6
                    n_785 = cauchy_A + cauchy_B / (785**2) * 1e6

                    # Clamp to physical range
                    cauchy_A = np.clip(cauchy_A, 1.5, 5.5)
                    n_1064 = np.clip(n_1064, 1.5, 6.0)
                    n_785 = np.clip(n_785, 1.5, 6.5)

                    records.append({
                        'metal': M, 'chalcogen': X, 'phase': phase_idx,
                        'atomic_number_M': Z_M, 'atomic_number_X': Z_X,
                        'electronegativity_M': chi_M, 'electronegativity_X': chi_X,
                        'atomic_radius_M': r_M, 'atomic_radius_X': r_X,
                        'ionization_energy_M': IE_M, 'electron_affinity_X': EA_X,
                        'd_electron_count': d_e,
                        'bond_length_MX': round(bond_length, 4),
                        'lattice_constant_a': round(a, 4),
                        'monolayer_thickness': round(thickness, 4),
                        'coordination_number': coord,
                        'point_group_encoded': pg,
                        'cauchy_A': round(cauchy_A, 6),
                        'cauchy_B': round(cauchy_B, 6),
                        'n_at_1064nm': round(n_1064, 6),
                        'n_at_785nm': round(n_785, 6),
                    })

        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} synthetic TMD samples")
        return df

    def _load_real_data(self) -> pd.DataFrame:
        """
        Load real TMD data via RealDataLoader.

        Source routing:
          'jarvis'          → JARVIS-DFT (pip install jarvis-tools, no download)
          'c2db'            → C2DB ASE db file  (requires self.c2db_path)
          'refractiveindex' → experimental YAML clone (requires self.ri_db_path)
          'auto'            → tries all in priority order
        """
        from src.real_data_loader import RealDataLoader
        loader = RealDataLoader()

        source = self.data_source
        logger.info(f"[DataPipeline] Loading real data: source='{source}'")

        try:
            if source == 'jarvis':
                df = loader.load_jarvis()
            elif source == 'c2db':
                if not self.c2db_path:
                    raise ValueError(
                        "c2db_path must be set when data_source='c2db'.\n"
                        "Example: --c2db-path /path/to/c2db.db"
                    )
                df = loader.load_c2db(self.c2db_path)
            elif source == 'refractiveindex':
                if not self.ri_db_path:
                    raise ValueError(
                        "ri_db_path must be set when data_source='refractiveindex'.\n"
                        "Clone: git clone "
                        "https://github.com/polyanskiy/refractiveindex.info-database"
                    )
                df = loader.load_refractiveindex(self.ri_db_path)
            elif source == 'csv':
                if not self.csv_path:
                    raise ValueError(
                        "csv_path must be set when data_source='csv'.\n"
                        "Example: --data-csv data/extended/tier2_binary.csv"
                    )
                df = loader.load_csv(self.csv_path)
            elif source == 'auto':
                df, used = loader.load_best_available(
                    c2db_path=self.c2db_path,
                    ri_db_path=self.ri_db_path,
                )
                logger.info(f"[DataPipeline] Auto-selected source: {used}")
            else:
                raise ValueError(
                    f"Unknown data_source='{source}'. "
                    "Choose: jarvis | c2db | refractiveindex | auto | synthetic"
                )

            if len(df) < 10:
                raise RuntimeError(
                    f"Only {len(df)} samples from '{source}' — too few to train. "
                    "Falling back to synthetic data."
                )
            logger.info(loader.summary(df))
            return df

        except Exception as e:
            logger.warning(
                f"[DataPipeline] Real data load failed ({e}). "
                "Falling back to synthetic data."
            )
            return self._generate_synthetic_tmd_data()

    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute physically motivated derived features."""
        df = df.copy()
        df['electronegativity_diff'] = abs(df['electronegativity_M'] - df['electronegativity_X'])
        df['radius_ratio'] = df['atomic_radius_M'] / df['atomic_radius_X']
        df['bond_polarity'] = ((df['electronegativity_X'] - df['electronegativity_M'])
                               / (df['electronegativity_X'] + df['electronegativity_M']))
        df['packing_fraction'] = (df['atomic_radius_M']**2 + 2 * df['atomic_radius_X']**2) / df['lattice_constant_a']**2
        df['Z_ratio'] = df['atomic_number_M'] / df['atomic_number_X']
        df['IE_EA_ratio'] = df['ionization_energy_M'] / df['electron_affinity_X']
        return df

    def _select_features(self) -> List[str]:
        """Select feature columns for symbolic regression."""
        structural = self.config['features']['structural']
        derived = self.config['features']['derived']
        # Also include computed derived features
        extra_derived = ['Z_ratio', 'IE_EA_ratio']
        all_features = structural + derived + extra_derived
        # Filter to columns actually present in dataframe
        return [f for f in all_features if f in self.df.columns]

    def get_feature_descriptions(self) -> dict:
        """Return human-readable descriptions of each feature for LLM prompts."""
        return {
            'atomic_number_M': 'Atomic number of the transition metal (Z_M)',
            'atomic_number_X': 'Atomic number of the chalcogen atom (Z_X)',
            'electronegativity_M': 'Pauling electronegativity of the metal (χ_M)',
            'electronegativity_X': 'Pauling electronegativity of the chalcogen (χ_X)',
            'atomic_radius_M': 'Atomic radius of the metal in Angstroms (r_M)',
            'atomic_radius_X': 'Atomic radius of the chalcogen in Angstroms (r_X)',
            'ionization_energy_M': 'First ionization energy of the metal in eV (IE_M)',
            'electron_affinity_X': 'Electron affinity of the chalcogen in eV (EA_X)',
            'd_electron_count': 'Number of d-electrons of the metal (d)',
            'bond_length_MX': 'M-X bond length in Angstroms (d_MX)',
            'lattice_constant_a': 'In-plane lattice constant in Angstroms (a)',
            'monolayer_thickness': 'Corrected monolayer thickness in Angstroms (t)',
            'coordination_number': 'Coordination number of metal atom (CN)',
            'point_group_encoded': 'Crystal point group (numerical encoding)',
            'electronegativity_diff': 'Electronegativity difference |χ_M - χ_X|',
            'radius_ratio': 'Radius ratio r_M / r_X',
            'bond_polarity': 'Bond polarity index (χ_X - χ_M)/(χ_X + χ_M)',
            'packing_fraction': 'Estimated atomic packing fraction',
            'Z_ratio': 'Atomic number ratio Z_M / Z_X',
            'IE_EA_ratio': 'Ionization energy to electron affinity ratio IE_M / EA_X',
        }
