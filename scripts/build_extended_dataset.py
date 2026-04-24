"""
Build extended 2D materials dataset for npj CM submission upgrade.

Creates three tiers from JARVIS-DFT 2D (latest, 1103 entries):
  Tier 1 : MX2 family (X in {O,S,Se,Te})               — ~87 entries
  Tier 2 : All M-X binaries (incl halides/pnictides)   — ~170 entries
  Tier 3 : All non-metallic 2D                         — ~553 entries

Each entry is annotated with:
  - n_static (Cauchy A proxy)      = sqrt((epsx + epsy) / 2)
  - bandgap, thickness, lattice a
  - 20 elemental / structural / derived descriptors

Usage:
    python scripts/build_extended_dataset.py
"""

import json
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from collections import Counter
from math import gcd
from pathlib import Path

# ---------------------------------------------------------------------------
# Extended element lookup (covers all elements that appear in JARVIS 2D data)
# Tuple = (Z, electronegativity_Pauling, atomic_radius_A, first_IE_eV, EA_eV, group)
# Data compiled from standard tables (CRC Handbook; WebElements).
# ---------------------------------------------------------------------------
ELEMENT_TABLE = {
    # H and light nonmetals
    'H':  (1,   2.20, 0.53, 13.60, 0.75),
    'Li': (3,   0.98, 1.82, 5.39,  0.62),
    'Be': (4,   1.57, 1.12, 9.32,  0.00),
    'B':  (5,   2.04, 0.87, 8.30,  0.28),
    'C':  (6,   2.55, 0.67, 11.26, 1.26),
    'N':  (7,   3.04, 0.56, 14.53, 0.07),
    'O':  (8,   3.44, 0.48, 13.62, 1.46),
    'F':  (9,   3.98, 0.42, 17.42, 3.40),
    # Period 3
    'Na': (11,  0.93, 1.90, 5.14,  0.55),
    'Mg': (12,  1.31, 1.45, 7.65,  0.00),
    'Al': (13,  1.61, 1.18, 5.99,  0.43),
    'Si': (14,  1.90, 1.11, 8.15,  1.39),
    'P':  (15,  2.19, 0.98, 10.49, 0.75),
    'S':  (16,  2.58, 0.88, 10.36, 2.08),
    'Cl': (17,  3.16, 0.79, 12.97, 3.61),
    # Period 4
    'K':  (19,  0.82, 2.43, 4.34,  0.50),
    'Ca': (20,  1.00, 1.94, 6.11,  0.02),
    'Sc': (21,  1.36, 1.84, 6.56,  0.19),
    'Ti': (22,  1.54, 1.76, 6.83,  0.08),
    'V':  (23,  1.63, 1.71, 6.75,  0.52),
    'Cr': (24,  1.66, 1.66, 6.77,  0.67),
    'Mn': (25,  1.55, 1.61, 7.43,  0.00),
    'Fe': (26,  1.83, 1.56, 7.90,  0.15),
    'Co': (27,  1.88, 1.52, 7.88,  0.66),
    'Ni': (28,  1.91, 1.49, 7.64,  1.16),
    'Cu': (29,  1.90, 1.45, 7.73,  1.24),
    'Zn': (30,  1.65, 1.42, 9.39,  0.00),
    'Ga': (31,  1.81, 1.36, 6.00,  0.43),
    'Ge': (32,  2.01, 1.25, 7.90,  1.23),
    'As': (33,  2.18, 1.14, 9.79,  0.81),
    'Se': (34,  2.55, 1.03, 9.75,  2.02),
    'Br': (35,  2.96, 0.94, 11.81, 3.36),
    # Period 5
    'Rb': (37,  0.82, 2.65, 4.18,  0.49),
    'Sr': (38,  0.95, 2.19, 5.69,  0.05),
    'Y':  (39,  1.22, 2.12, 6.22,  0.31),
    'Zr': (40,  1.33, 2.06, 6.63,  0.43),
    'Nb': (41,  1.60, 1.98, 6.76,  0.89),
    'Mo': (42,  2.16, 1.90, 7.09,  0.75),
    'Tc': (43,  1.90, 1.83, 7.28,  0.55),
    'Ru': (44,  2.20, 1.78, 7.36,  1.05),
    'Rh': (45,  2.28, 1.73, 7.46,  1.14),
    'Pd': (46,  2.20, 1.69, 8.34,  0.56),
    'Ag': (47,  1.93, 1.65, 7.58,  1.30),
    'Cd': (48,  1.69, 1.61, 8.99,  0.00),
    'In': (49,  1.78, 1.56, 5.79,  0.30),
    'Sn': (50,  1.96, 1.45, 7.34,  1.11),
    'Sb': (51,  2.05, 1.33, 8.61,  1.05),
    'Te': (52,  2.10, 1.23, 9.01,  1.97),
    'I':  (53,  2.66, 1.15, 10.45, 3.06),
    # Period 6 (lanthanides + heavy)
    'Cs': (55,  0.79, 2.98, 3.89,  0.47),
    'Ba': (56,  0.89, 2.53, 5.21,  0.14),
    'La': (57,  1.10, 2.26, 5.58,  0.47),
    'Ce': (58,  1.12, 2.10, 5.54,  0.65),
    'Pr': (59,  1.13, 2.47, 5.47,  0.96),
    'Nd': (60,  1.14, 2.06, 5.53,  1.92),
    'Sm': (62,  1.17, 2.38, 5.64,  0.13),
    'Eu': (63,  1.20, 2.31, 5.67,  0.86),
    'Gd': (64,  1.20, 2.33, 6.15,  0.14),
    'Tb': (65,  1.20, 2.25, 5.86,  1.17),
    'Dy': (66,  1.22, 2.28, 5.94,  0.35),
    'Ho': (67,  1.23, 2.26, 6.02,  0.34),
    'Er': (68,  1.24, 2.26, 6.11,  0.31),
    'Tm': (69,  1.25, 2.22, 6.18,  1.03),
    'Yb': (70,  1.10, 2.22, 6.25,  0.02),
    'Lu': (71,  1.27, 2.17, 5.43,  0.34),
    'Hf': (72,  1.30, 2.08, 6.83,  0.00),
    'Ta': (73,  1.50, 2.00, 7.55,  0.32),
    'W':  (74,  2.36, 1.93, 7.86,  0.82),
    'Re': (75,  1.90, 1.88, 7.83,  0.15),
    'Os': (76,  2.20, 1.85, 8.44,  1.10),
    'Ir': (77,  2.20, 1.80, 8.97,  1.56),
    'Pt': (78,  2.28, 1.77, 8.96,  2.13),
    'Au': (79,  2.54, 1.74, 9.23,  2.31),
    'Hg': (80,  2.00, 1.71, 10.44, 0.00),
    'Tl': (81,  1.62, 1.56, 6.11,  0.38),
    'Pb': (82,  2.33, 1.54, 7.42,  0.36),
    'Bi': (83,  2.02, 1.43, 7.29,  0.94),
}

ANION_FAMILIES = {
    'chalcogen': {'O', 'S', 'Se', 'Te'},
    'halogen':   {'F', 'Cl', 'Br', 'I'},
    'pnictogen': {'N', 'P', 'As', 'Sb'},
}

METALS_AND_METALLOIDS = {
    'Li','Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga',
    'Ge','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb',
    'Cs','Ba','La','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi',
    'Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
    'As','B','Si',
}

D_ELECTRONS = {
    'Sc':1,'Ti':2,'V':3,'Cr':5,'Mn':5,'Fe':6,'Co':7,'Ni':8,'Cu':10,'Zn':10,
    'Y':1,'Zr':2,'Nb':4,'Mo':5,'Tc':5,'Ru':7,'Rh':8,'Pd':10,'Ag':10,'Cd':10,
    'Hf':2,'Ta':3,'W':4,'Re':5,'Os':6,'Ir':7,'Pt':9,'Au':10,'Hg':10,
}

# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------
def is_valid_optical(e):
    for k in ['epsx', 'epsy', 'epsz']:
        v = e.get(k)
        if not isinstance(v, (int, float)):
            return False
    if min(e['epsx'], e['epsy']) < 1.0 or max(e['epsx'], e['epsy']) > 50:
        return False
    gap = e.get('optb88vdw_bandgap')
    if not isinstance(gap, (int, float)) or gap < 0.1:
        return False
    return True

def classify_tier(atoms_dict):
    """Return (tier, metal_symbol, anion_symbol, stoichiometry) or None."""
    atoms = Atoms.from_dict(atoms_dict)
    elements = atoms.elements
    elem_set = set(elements)
    counter = Counter(elements)

    # Tier 3: any non-metallic 2D
    result = {'tier': 3, 'metal': None, 'anion': None, 'n_metal': 0, 'n_anion': 0}

    if len(elem_set) != 2:
        return result

    metals = elem_set & METALS_AND_METALLOIDS
    nonmetals = elem_set & (ANION_FAMILIES['chalcogen'] |
                            ANION_FAMILIES['halogen'] |
                            ANION_FAMILIES['pnictogen'])
    if len(metals) != 1 or len(nonmetals) != 1:
        return result

    m = list(metals)[0]
    a = list(nonmetals)[0]
    n_m = counter[m]; n_a = counter[a]
    g = gcd(n_m, n_a)
    n_m_r, n_a_r = n_m // g, n_a // g

    result.update({'tier': 2, 'metal': m, 'anion': a,
                   'n_metal': n_m_r, 'n_anion': n_a_r})

    # Tier 1: MX2 with X in {O, S, Se, Te}
    if n_m_r == 1 and n_a_r == 2 and a in ANION_FAMILIES['chalcogen']:
        result['tier'] = 1

    return result

# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
def compute_features(metal, anion, atoms_dict, n_metal, n_anion):
    atoms = Atoms.from_dict(atoms_dict)
    lat = atoms.lattice_mat  # 3x3
    a_lat = np.linalg.norm(lat[0])

    # Thickness estimation: out-of-plane extent of fractional coords
    coords = np.array(atoms.frac_coords)
    z_coords = coords[:, 2]
    c_vec_length = np.linalg.norm(lat[2])
    z_extent = (z_coords.max() - z_coords.min()) * c_vec_length
    thickness = max(z_extent, 1.0)  # avoid zero

    # Bond length estimate: nearest M-X in Cartesian
    cart = np.array(atoms.cart_coords)
    el_list = atoms.elements
    m_idx = [i for i, e in enumerate(el_list) if e == metal]
    a_idx = [i for i, e in enumerate(el_list) if e == anion]
    if m_idx and a_idx:
        dmin = 1e9
        for i in m_idx:
            for j in a_idx:
                d = np.linalg.norm(cart[i] - cart[j])
                if 0.5 < d < dmin:
                    dmin = d
        d_MX = dmin if dmin < 1e9 else np.nan
    else:
        d_MX = np.nan

    Z_M, chi_M, r_M, IE_M, EA_M = ELEMENT_TABLE[metal]
    Z_X, chi_X, r_X, IE_X, EA_X = ELEMENT_TABLE[anion]

    d_e = D_ELECTRONS.get(metal, 0)

    feat = {
        'metal': metal,
        'anion': anion,
        'stoichiometry': f"M{n_metal if n_metal>1 else ''}X{n_anion if n_anion>1 else ''}",
        # Elemental
        'Z_M': Z_M,  'Z_X': Z_X,
        'chi_M': chi_M, 'chi_X': chi_X,
        'r_M': r_M, 'r_X': r_X,
        'IE_M': IE_M, 'EA_X': EA_X,
        'd_e': d_e,
        # Structural
        'd_MX': d_MX,
        'a_lat': a_lat,
        't': thickness,
        'n_M': n_metal,
        'n_X': n_anion,
        'stoich_ratio': n_anion / n_metal,
        # Derived
        'abs_delta_chi': abs(chi_X - chi_M),
        'r_ratio': r_M / r_X,
        'bond_polarity': (chi_X - chi_M) / (chi_X + chi_M),
        'Z_ratio': Z_M / Z_X,
        'IE_EA_ratio': IE_M / EA_X if EA_X > 0.01 else np.nan,
    }
    return feat

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    raw_path = Path('data/extended/jarvis_dft2d_raw.json')
    with open(raw_path) as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} raw JARVIS 2D entries")

    records = []
    skipped_elements = Counter()

    for e in raw:
        if not is_valid_optical(e):
            continue

        cls = classify_tier(e['atoms'])

        # For Tier 2/1 we can compute full features
        if cls['tier'] in (1, 2):
            metal = cls['metal']; anion = cls['anion']
            if metal not in ELEMENT_TABLE or anion not in ELEMENT_TABLE:
                skipped_elements[metal] += 1
                continue
            try:
                feat = compute_features(metal, anion, e['atoms'],
                                         cls['n_metal'], cls['n_anion'])
            except Exception as ex:
                continue

            # DFT-derived properties
            eps_xx = e['epsx']; eps_yy = e['epsy']; eps_zz = e['epsz']
            eps_inplane = (eps_xx + eps_yy) / 2
            n_static = np.sqrt(eps_inplane)
            A_cauchy = n_static

            feat.update({
                'jid': e['jid'],
                'formula': e['formula'],
                'tier': cls['tier'],
                'bandgap_optb88': e['optb88vdw_bandgap'],
                'eps_xx': eps_xx,
                'eps_yy': eps_yy,
                'eps_zz': eps_zz,
                'eps_inplane': eps_inplane,
                'n_static': n_static,
                'cauchy_A': A_cauchy,
                'spg_number': e.get('spg_number'),
            })
            records.append(feat)

    df = pd.DataFrame(records)
    print(f"\nBuilt dataset: {df.shape}")
    print(f"Tier 1 (MX2 chalcogenide): {(df['tier']==1).sum()}")
    print(f"Tier 2 (all binary M-X): {(df['tier']==2).sum()}")
    print(f"Stoichiometry distribution:")
    print(df['stoichiometry'].value_counts().head(20))
    print(f"\nAnion distribution:")
    print(df['anion'].value_counts())
    print(f"\nCauchy A stats: mean={df['cauchy_A'].mean():.3f}, "
          f"std={df['cauchy_A'].std():.3f}, "
          f"min={df['cauchy_A'].min():.3f}, max={df['cauchy_A'].max():.3f}")

    # Save each tier as its own file
    out_dir = Path('data/extended')
    df_t1 = df[df['tier'] == 1].reset_index(drop=True)
    df_t2 = df[df['tier'].isin([1, 2])].reset_index(drop=True)

    df_t1.to_csv(out_dir / 'tier1_mx2.csv', index=False)
    df_t2.to_csv(out_dir / 'tier2_binary.csv', index=False)
    print(f"\nSaved:")
    print(f"  Tier 1 MX2:     data/extended/tier1_mx2.csv     ({len(df_t1)} samples)")
    print(f"  Tier 2 Binary:  data/extended/tier2_binary.csv  ({len(df_t2)} samples)")

    if skipped_elements:
        print(f"\nSkipped unknown elements: {dict(skipped_elements)}")

if __name__ == '__main__':
    main()
