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
        # Single-spectrum splits of the same plik-lite data (diagnostic only)
        "plik_lite_TT": {"class": "planck_2018_highl_plik.TTTEEE_lite_native", "dataset_params": {"use_cl": "tt"}},
        "plik_lite_TE": {"class": "planck_2018_highl_plik.TTTEEE_lite_native", "dataset_params": {"use_cl": "te"}},
        "plik_lite_EE": {"class": "planck_2018_highl_plik.TTTEEE_lite_native", "dataset_params": {"use_cl": "ee"}},
    },
    # Planck 2018 TT,TE,EE+lowE+lensing posterior means (Planck 2018 VI, Table 2)
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

like_keys = list(model.likelihood)
split_keys = {"plik_lite_TT", "plik_lite_TE", "plik_lite_EE"}
in_total = [k not in split_keys for k in like_keys]   # splits excluded from the total

def chi2_at(A_d, ell_d):
    """Run CLASS + likelihoods at one point; return total chi2 and per-likelihood chi2.
    The total excludes the TT/TE/EE splits (they'd double-count the TTTEEE data)."""
    lp = model.logposterior({"thomson_rescaling": A_d,
                             "phenomenological_damping_tail": ell_d})
    per_like = [-2 * ll for ll in lp.loglikes]
    total = sum(c for c, use in zip(per_like, in_total) if use)
    return total, per_like

names = [n.split(".")[-1] for n in like_keys]
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
np.savetxt("Planck_scan_A_d.txt", rows, header=header)
 
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
np.savetxt("Planck_scan_ell_d.txt", rows, header=header)