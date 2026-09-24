from cobaya.model import get_model
import numpy as np

################################################################################ Model info

'''
# Run in CLASS_SilkConsistency:
cobaya-install planck_2018_highl_plik.TTTEEE_lite_native planck_2018_lowl.TT \ 
        planck_2018_lowl.EE planck_2018_lensing.native -p ./packages
'''
info = {
    "packages_path": "../packages", 
    "theory": {"classy": {"path": "global",
                          "extra_args": {"N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": 0.06}}},
    "likelihood": {
        "planck_2018_lowl.TT": None,
        "planck_2018_lowl.EE": None,
        "planck_2018_highl_plik.TTTEEE_lite_native": None,
        "planck_2018_lensing.native": None,
    },
    # Planck 2018 best fit
    "params": {
        "omega_b": 0.02237, "omega_cdm": 0.1200, "H0": 67.36,
        "ln10^{10}A_s": 3.044, "n_s": 0.9649, "tau_reio": 0.0544,
        "A_planck": 1.0,
        "thomson_rescaling": {"prior": {"min": 0.5, "max": 1.5}},              # A_d
        "phenomenological_damping_tail": {"prior": {"min": 0., "max": 1e6}},  # l_d (0 = off)
    },
}
model = get_model(info)

################################################################################ Likelihood

def chi2_at(A_d, ell_d):
    """Run CLASS + likelihoods at one point; return total chi2 and per-likelihood chi2."""
    lp = model.logposterior({"thomson_rescaling": A_d,
                             "phenomenological_damping_tail": ell_d})
    per_like = [-2 * ll for ll in lp.loglikes]
    return sum(per_like), per_like
names = [n.split(".")[-1] for n in model.likelihood]
header = "value chi2_total " + " ".join(names)

################################################################################ Running 1D scans
 
A_d_grid = np.unique(np.concatenate((
    np.linspace(0.98, 1.02, 41),        # step 0.001, resolves the peak, includes 1.0
    [0.8, 0.9, 0.95, 1.05, 1.1, 1.2],   # a few far points for context
)))
rows = []
for A in A_d_grid:
    tot, per = chi2_at(A, 0.)
    rows.append([A, tot, *per])
    print(f"A_d = {A:.4f}   chi2 = {tot:.3f}")
np.savetxt("scan_A_d.txt", rows, header=header)
 
ell_d_grid = np.unique(np.concatenate((
    [0.],                                           # LCDM (envelope off)
    np.logspace(np.log10(5e4), np.log10(6e3), 25),  # dense: where the curve changes
    [1e5, 2e5, 5e5, 1e6],                           # far out: should converge to LCDM
    [2e3, 3e3, 4e3],                                # far in: strongly excluded, for context
)))
rows = []
for ld in ell_d_grid:
    tot, per = chi2_at(1.0, ld)
    rows.append([ld, tot, *per])
    print(f"ell_d = {ld:7.0f}   chi2 = {tot:.3f}")
np.savetxt("scan_ell_d.txt", rows, header=header)