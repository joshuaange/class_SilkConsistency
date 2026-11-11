"""
Make a noiseless mock of an EARLY DARK ENERGY universe, for the injection-recovery test,
using CLASS_EDE (Hill, McDonough, Toomey & Alexander 2020, arXiv:2003.07355;
https://github.com/mwt5345/class_ede).

Your CLASS fork has no EDE, so the EDE effect is computed with CLASS_EDE and added on top of
the CLASS LCDM mock you already made with make_mock.py:

    mock_EDE = C_yourCLASS(LCDM)  +  [ C_CLASS_EDE(EDE) - C_CLASS_EDE(LCDM) ]

Only the DIFFERENCE comes from CLASS_EDE, so small differences between the two CLASS versions
cancel to first order. The covariance is the same LCDM one used for the other mocks.

RUN THIS IN THE SEPARATE CLASS_EDE ENVIRONMENT (its Python module is also called `classy`):
    source ~/envs/class_ede/bin/activate
    python make_mock_ede.py
"""
import numpy as np
from classy import Class

LMAX = 4000          # same as make_mock.py

# ---- settings shared by both CLASS_EDE runs (as close to your old mock settings as possible) ----
COMMON = {
    "output": "tCl pCl lCl", "lensing": "yes", "l_max_scalars": LMAX,
    "non_linear": "hmcode",
    "N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": 0.06,     # same neutrinos as make_mock.py
}

# ---- 1. LCDM: the SAME fiducial cosmology as make_mock.py's LCDM mock ----
LCDM = {"omega_b": 0.02237, "omega_cdm": 0.1200, "100*theta_s": 1.04190,
        "ln10^{10}A_s": 3.044, "n_s": 0.9649, "tau_reio": 0.0544}

# ---- 2. EDE: best fit of EDE (n=3) to ACT DR4 TT+TE+EE + Planck 2018 TT (ell<650) + Planck
#         lensing + BAO + tau prior. Hill et al. 2022, PRD 105, 123536 (arXiv:2109.04451),
#         Table VI, best-fit EDE column.
EDE = {"fEDE": 0.091, "log10z_c": 3.16, "thetai_scf": 0.72,   # EDE parameters (theta_i = phi_i/f)
       "h": 0.709, "omega_b": 0.02120, "omega_cdm": 0.1289,
       "ln10^{10}A_s": 3.050, "n_s": 0.9684, "tau_reio": 0.055,
       # how CLASS_EDE switches the scalar field on (from its README / example notebooks):
       "n_scf": 3, "CC_scf": 1, "Omega_Lambda": 0.0, "Omega_fld": 0, "Omega_scf": -1,
       "scf_parameters": "1, 1, 1, 1, 1, 0.0", "scf_tuning_index": 3, "attractor_ic_scf": "no"}


def class_ede_cls(params):
    M = Class()
    M.set({**COMMON, **params})
    M.compute()
    cl = M.lensed_cl(LMAX)
    T2 = (M.T_cmb() * 1e6) ** 2                       # dimensionless -> muK^2
    out = np.array([cl["tt"], cl["ee"], cl["te"]]) * T2
    info = {"H0": M.h() * 100}
    if "fEDE" in params:
        info.update(fEDE=M.fEDE(), log10z_c=M.log10z_c())
    M.struct_cleanup(); M.empty()
    return out, info


C_lcdm, info_lcdm = class_ede_cls(LCDM)
C_ede, info_ede = class_ede_cls(EDE)
dC = C_ede - C_lcdm

# ---- 3. add the difference to your CLASS LCDM mock (same ells, same covariance) ----
base = np.load("mock_LCDM.npz")
ell = base["ell"]
data = base["data"] + dC[:, ell].T
np.savez("mock_EDE.npz", ell=ell, data=data, icov=base["icov"])

print(f"mock_EDE.npz written   (true H0 = {info_ede['H0']:.2f}; LCDM baseline {info_lcdm['H0']:.2f})")
print(f"  EDE achieved by CLASS_EDE: fEDE = {info_ede['fEDE']:.4f} (target 0.091), "
      f"log10 z_c = {info_ede['log10z_c']:.3f} (target 3.16)")
for l in (1000, 2000, 3000):
    i = l - ell[0]
    print(f"  ell={l}:  TT {100*(data[i,0]/base['data'][i,0]-1):+.2f}%   EE {100*(data[i,1]/base['data'][i,1]-1):+.2f}%")
