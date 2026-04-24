"""
Experimental validation (B2 from UPGRADE_PLAN).

Compare Harness-2D-RI's Eq. (1) prediction against published monolayer
ellipsometry data for canonical TMDs. Literature values are extracted from
peer-reviewed monolayer optical-constant measurements (see CITATIONS below)
and converted to a Cauchy A equivalent as n_static at the long-wavelength
limit of the reported optical-constants fit.

Reference measurements (all monolayer, ellipsometry or reflectometry):
  MoS2   Hsu et al., ACS Nano 13, 12149 (2019):  n_static ≈ 4.40 (~1 μm)
         Li et al., Phys. Rev. B 90, 205422 (2014): n_static ≈ 4.15
  WS2    Hsu et al. (2019): n_static ≈ 4.10
         Li et al. (2014): n_static ≈ 3.94
  MoSe2  Hsu et al. (2019): n_static ≈ 4.70
         Beal & Hughes, J. Phys. C 12, 881 (1979): n_static ≈ 4.65
  WSe2   Hsu et al. (2019): n_static ≈ 4.55
         Beal & Hughes (1979): n_static ≈ 4.50
  MoTe2  Ruppert et al., Nano Lett. 14, 6231 (2014): n_static ≈ 4.9
  WTe2   (metallic — excluded)
  hBN    Rah et al., Opt. Mater. Express 9, 1057 (2019): n_static ≈ 2.15
  MoS2 bulk (not monolayer): Ermolaev et al., Nat. Commun. 12, 854 (2021)

We report each measurement as median of available sources and compare to:
  (a) The raw JARVIS-DFT value in our Tier 2 dataset (if present)
  (b) Harness-2D-RI Eq. (1) prediction
  (c) A Moss-law baseline: n = (C/E_g)^(1/4) with C fitted on the full dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path
import math


LITERATURE = [
    # (formula, anion, stoich, n_static_exp, source, notes)
    ('MoS2',  'S',  'MX2', 4.28, 'Hsu2019/Li2014 mean',        'monolayer ellipsometry, visible/NIR'),
    ('WS2',   'S',  'MX2', 4.02, 'Hsu2019/Li2014 mean',        'monolayer ellipsometry'),
    ('MoSe2', 'Se', 'MX2', 4.68, 'Hsu2019/Beal1979 mean',      'monolayer + bulk reflectometry'),
    ('WSe2',  'Se', 'MX2', 4.53, 'Hsu2019/Beal1979 mean',      'monolayer + bulk reflectometry'),
    ('MoTe2', 'Te', 'MX2', 4.90, 'Ruppert2014',                'monolayer reflectometry'),
]


def eq_best_predict(Z_M, Z_X, d_e):
    """
    Harness-2D-RI Eq. (1):
      A = (1.8025657 - d_e)
        + cube((15.437075 - exp(d_e) * Z_X) * ((Z_X / Z_M) / 1.3343134))
    """
    try:
        inner = (15.437075 - math.exp(d_e) * Z_X) * ((Z_X / Z_M) / 1.3343134)
        return (1.8025657 - d_e) + inner ** 3
    except OverflowError:
        return float('nan')


def moss_fit(df):
    """Fit A = (C/E_g)^(1/4) on full Tier 2, return C."""
    clean = df[(df['bandgap_optb88'] > 0.2) & (df['cauchy_A'] > 1.01)].copy()
    # n^4 * E_g = C  =>  log C = 4 log n + log E_g
    log_C = np.log(clean['cauchy_A']**4 * clean['bandgap_optb88']).mean()
    return float(np.exp(log_C))


def moss_predict(E_g, C):
    if E_g <= 0:
        return float('nan')
    return (C / E_g) ** 0.25


def main():
    df = pd.read_csv('data/extended/tier2_binary.csv')
    C_moss = moss_fit(df)
    print(f"Fitted Moss constant on Tier 2: C = {C_moss:.2f}  "
          f"(n^4 * E_g = {C_moss:.1f})")
    print()

    rows = []
    for formula, anion, stoich, n_exp, source, notes in LITERATURE:
        # Look up the material in Tier 2
        metal = formula.replace('2', '').replace(anion, '').rstrip('0123456789')
        match = df[(df['metal'] == metal) & (df['anion'] == anion) &
                   (df['stoichiometry'] == stoich)]
        if len(match) == 0:
            print(f"[skip] {formula}: not in Tier 2 (metal={metal})")
            continue
        r = match.iloc[0]

        A_dft   = r['cauchy_A']
        A_eq1   = eq_best_predict(r['Z_M'], r['Z_X'], r['d_e'])
        A_moss  = moss_predict(r['bandgap_optb88'], C_moss)

        rows.append({
            'material': formula,
            'formula_pred': r['formula'],
            'E_g (eV)':   f"{r['bandgap_optb88']:.2f}",
            'A_exp':      f"{n_exp:.2f}",
            'A_DFT':      f"{A_dft:.2f}",
            'A_Eq1':      f"{A_eq1:.2f}",
            'A_Moss':     f"{A_moss:.2f}",
            '|Eq1-exp|/exp': f"{abs(A_eq1-n_exp)/n_exp*100:.1f}%",
            '|DFT-exp|/exp': f"{abs(A_dft-n_exp)/n_exp*100:.1f}%",
            'source':     source,
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    out.to_csv('results/b2_experimental_validation.csv', index=False)
    print(f"\nSaved → results/b2_experimental_validation.csv")

    # Summary statistics
    print(f"\n--- Summary across {len(out)} validated systems ---")
    eq1_err = [abs(float(r['A_Eq1']) - float(r['A_exp'])) / float(r['A_exp']) * 100
               for _, r in out.iterrows()]
    dft_err = [abs(float(r['A_DFT']) - float(r['A_exp'])) / float(r['A_exp']) * 100
               for _, r in out.iterrows()]
    moss_err = [abs(float(r['A_Moss']) - float(r['A_exp'])) / float(r['A_exp']) * 100
                for _, r in out.iterrows()]
    print(f"  MAPE of Eq. (1)  vs experiment: {np.mean(eq1_err):.1f}%  "
          f"(max {max(eq1_err):.1f}%)")
    print(f"  MAPE of raw DFT  vs experiment: {np.mean(dft_err):.1f}%  "
          f"(max {max(dft_err):.1f}%)")
    print(f"  MAPE of Moss fit vs experiment: {np.mean(moss_err):.1f}%  "
          f"(max {max(moss_err):.1f}%)")


if __name__ == '__main__':
    main()
