"""
Make noiseless ("Asimov") mock CMB spectra for an injection-recovery test.

  mock_LCDM.npz : the Planck best-fit LCDM          (control: should recover A_d = 1, eps_d = 0)
  mock_Neff.npz : the same + 0.5 extra N_eff        (H0 ~ 70.4)

The spectra are computed THROUGH COBAYA, with the same theory settings the
MCMC uses, so a fit at the true parameters gives chi2 = 0 exactly.
Run it in the same environment you run the MCMCs in.
"""
import numpy as np
from cobaya.model import get_model

# ---- the experiment (edit these to change the noise level) ----------------
LMIN, LMAX = 2, 4000
NOISE_T = 15.0     # white noise in temperature, muK-arcmin (polarization: x sqrt 2)
BEAM = 1.4         # beam FWHM, arcmin
FSKY = 0.4

# ---- the models ------------------------------------------------------------
N_UR_STD = 2.0328                       # + 1 massive nu (0.06 eV) -> N_eff = 3.044
DN = 0.5                                # extra N_eff in the mock
FAC = (1 + 0.2271 * (3.044 + DN)) / (1 + 0.2271 * 3.044)   # rescales omega_cdm to keep z_eq fixed
lcdm = {"omega_b": 0.02237, "omega_cdm": 0.1200, "100*theta_s": 1.04190,
        "ln10^{10}A_s": 3.044, "n_s": 0.9649, "tau_reio": 0.0544}
mocks = {"LCDM": (N_UR_STD, lcdm),
         "Neff": (N_UR_STD + DN, {**lcdm, "omega_cdm": lcdm["omega_cdm"] * FAC})}


def lensed_cls(n_ur, params):
    """Lensed TT, EE, TE (muK^2) and H0, from the same classy setup as the MCMC yamls."""
    info = {"theory": {"classy": {"path": "global",
                                  "extra_args": {"N_ur": n_ur, "N_ncdm": 1, "m_ncdm": 0.06}}},
            "likelihood": {"one": None},                     # dummy: we only want the theory
            "params": {**params, "H0": {"latex": "H_0"}}}       # H0 as a derived output
    model = get_model(info)
    model.add_requirements({"Cl": {"tt": LMAX, "ee": LMAX, "te": LMAX}})
    lp = model.logposterior({})
    H0 = dict(zip(model.parameterization.derived_params(), lp.derived))["H0"]
    cl = model.provider.get_Cl(ell_factor=False, units="muK2")
    return np.array([cl["tt"], cl["ee"], cl["te"]]), H0


ell = np.arange(LMIN, LMAX + 1)

# ---- covariance: Gaussian, from the LCDM model + noise, the same for both mocks
C, _ = lensed_cls(*mocks["LCDM"])
th = BEAM / 60 * np.pi / 180
NT = (NOISE_T * np.pi / 10800) ** 2 * np.exp(ell * (ell + 1) * th ** 2 / (8 * np.log(2)))
tt, ee, te = C[0, ell] + NT, C[1, ell] + 2 * NT, C[2, ell]
cov = np.zeros((len(ell), 3, 3))                # order: TT, EE, TE
cov[:, 0, 0] = 2 * tt ** 2
cov[:, 1, 1] = 2 * ee ** 2
cov[:, 2, 2] = te ** 2 + tt * ee
cov[:, 0, 1] = cov[:, 1, 0] = 2 * te ** 2
cov[:, 0, 2] = cov[:, 2, 0] = 2 * tt * te
cov[:, 1, 2] = cov[:, 2, 1] = 2 * ee * te
cov /= ((2 * ell + 1) * FSKY)[:, None, None]
icov = np.linalg.inv(cov)

# ---- the mock data vectors
for name, (n_ur, p) in mocks.items():
    D, H0 = lensed_cls(n_ur, p)
    np.savez(f"mock_{name}.npz", ell=ell, data=D[:, ell].T, icov=icov)
    print(f"mock_{name}.npz written   (true H0 = {H0:.2f})")
