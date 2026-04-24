"""
Build Tier 3 expanded dataset for npj CM reviewer-stress-test.

Strategy: take all 544 JARVIS-DFT2D semiconductors with valid optical, then
assign (M, X) = (most_electropositive, most_electronegative) for each entry
regardless of element count. Ternaries use their 'primary' cation/anion pair
plus three ternary-specific features. Binaries reuse Tier-2 featurisation.

Output: data/extended/tier3_binary_plus_ternary.csv
"""

import json
import numpy as np
import pandas as pd
from collections import Counter
from math import gcd
from pathlib import Path
from jarvis.core.atoms import Atoms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_extended_dataset import ELEMENT_TABLE, D_ELECTRONS


def is_valid_optical(e):
    for k in ['epsx', 'epsy']:
        v = e.get(k)
        if not isinstance(v, (int, float)):
            return False
    if min(e['epsx'], e['epsy']) < 1.0 or max(e['epsx'], e['epsy']) > 50:
        return False
    gap = e.get('optb88vdw_bandgap')
    if not isinstance(gap, (int, float)) or gap < 0.2:
        return False
    return True


def primary_MX(elements):
    """Return (metal, anion) = (most electropositive, most electronegative) by Pauling χ."""
    present = [el for el in set(elements) if el in ELEMENT_TABLE]
    if len(present) < 2:
        return None, None
    by_chi = sorted(present, key=lambda el: ELEMENT_TABLE[el][1])
    return by_chi[0], by_chi[-1]


def featurise(e, metal, anion):
    atoms = Atoms.from_dict(e['atoms'])
    elements = atoms.elements
    counter = Counter(elements)
    n_metal = counter.get(metal, 0)
    n_anion = counter.get(anion, 0)
    n_total = len(elements)
    n_elem = len(set(elements))
    if n_metal == 0 or n_anion == 0:
        return None

    g = gcd(n_metal, n_anion)
    n_m_r, n_a_r = n_metal // g, n_anion // g

    lat = atoms.lattice_mat
    a_lat = float(np.linalg.norm(lat[0]))
    coords = np.array(atoms.frac_coords)
    c_len = float(np.linalg.norm(lat[2]))
    thickness = max((coords[:, 2].max() - coords[:, 2].min()) * c_len, 1.0)

    cart = np.array(atoms.cart_coords)
    m_idx = [i for i, el in enumerate(elements) if el == metal]
    a_idx = [i for i, el in enumerate(elements) if el == anion]
    dmin = 1e9
    for i in m_idx:
        for j in a_idx:
            d = float(np.linalg.norm(cart[i] - cart[j]))
            if 0.5 < d < dmin:
                dmin = d
    d_MX = dmin if dmin < 1e9 else np.nan

    Z_M, chi_M, r_M, IE_M, _ = ELEMENT_TABLE[metal]
    Z_X, chi_X, r_X, IE_X, EA_X = ELEMENT_TABLE[anion]
    d_e = D_ELECTRONS.get(metal, 0)

    feat = {
        'metal': metal, 'anion': anion,
        'stoichiometry': f"M{n_m_r if n_m_r > 1 else ''}X{n_a_r if n_a_r > 1 else ''}",
        'Z_M': Z_M, 'Z_X': Z_X,
        'chi_M': chi_M, 'chi_X': chi_X,
        'r_M': r_M, 'r_X': r_X,
        'IE_M': IE_M, 'EA_X': EA_X,
        'd_e': d_e,
        'd_MX': d_MX, 'a_lat': a_lat, 't': thickness,
        'n_M': n_metal, 'n_X': n_anion,
        'stoich_ratio': n_anion / n_metal if n_metal else np.nan,
        'abs_delta_chi': abs(chi_X - chi_M),
        'r_ratio': r_M / r_X if r_X else np.nan,
        'bond_polarity': (chi_X - chi_M) / (chi_X + chi_M) if (chi_X + chi_M) else np.nan,
        'Z_ratio': Z_M / Z_X if Z_X else np.nan,
        'IE_EA_ratio': IE_M / EA_X if EA_X > 0.01 else np.nan,
        # Ternary/complexity indicators
        'n_elem': n_elem,
        'n_atoms': n_total,
        'is_binary': int(n_elem == 2),
    }
    return feat


def main():
    raw = json.load(open('data/extended/jarvis_dft2d_raw.json'))
    print(f"Loaded {len(raw)} JARVIS-DFT2D entries")

    records = []
    skipped = Counter()

    for e in raw:
        if not is_valid_optical(e):
            continue
        atoms = Atoms.from_dict(e['atoms'])
        metal, anion = primary_MX(atoms.elements)
        if metal is None:
            skipped['no_MX_pair'] += 1
            continue
        try:
            feat = featurise(e, metal, anion)
        except Exception as ex:
            skipped['featurise_err'] += 1
            continue
        if feat is None:
            skipped['no_counts'] += 1
            continue

        eps_xx = e['epsx']
        eps_yy = e['epsy']
        eps_inplane = (eps_xx + eps_yy) / 2
        feat.update({
            'jid': e['jid'],
            'formula': e['formula'],
            'tier': 3,
            'bandgap_optb88': e['optb88vdw_bandgap'],
            'eps_xx': eps_xx, 'eps_yy': eps_yy,
            'eps_zz': e.get('epsz'),
            'eps_inplane': eps_inplane,
            'n_static': np.sqrt(eps_inplane),
            'cauchy_A': np.sqrt(eps_inplane),
        })
        records.append(feat)

    df = pd.DataFrame(records)
    print(f"\nBuilt Tier 3: {df.shape}")
    print(f"Binary (n_elem=2): {(df['n_elem']==2).sum()}")
    print(f"Ternary (n_elem=3): {(df['n_elem']==3).sum()}")
    print(f"Quaternary+ (n_elem>=4): {(df['n_elem']>=4).sum()}")
    print(f"Unique metals: {df['metal'].nunique()}; anions: {df['anion'].nunique()}")
    print(f"Anion distribution:")
    print(df['anion'].value_counts().to_string())
    print(f"\nCauchy A stats: mean={df['cauchy_A'].mean():.3f} "
          f"std={df['cauchy_A'].std():.3f} "
          f"range=[{df['cauchy_A'].min():.2f}, {df['cauchy_A'].max():.2f}]")

    out = Path('data/extended/tier3_binary_plus_ternary.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out} ({len(df)} samples)")
    if skipped:
        print(f"Skipped: {dict(skipped)}")


if __name__ == '__main__':
    main()
