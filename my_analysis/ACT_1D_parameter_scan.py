from cobaya.model import get_model
import numpy as np

################################################################################ Model info

'''
# Run in CLASS_SilkConsistency:
git clone https://github.com/ACTCollaboration/DR6-ACT-lite.git
pip install -e DR6-ACT-lite
cobaya-install act_dr6_cmbonly -p ./packages
'''
act_name = "act_dr6_cmbonly.ACTDR6CMBonly"
info = {
    "packages_path": "../packages",
    "theory": {"classy": {"path": "global",
                          "extra_args": {"N_ur": 2.0328, "N_ncdm": 1, "m_ncdm": 0.06}}},
    "likelihood": {
        act_name: None,   # ACT DR6 TT+TE+EE, foreground-marginalized ("ACT-lite"), 600 < ell < 6500
    },
    # ACT DR6 LCDM (Louis et al. 2025, Table 5)
    "params": {
        "omega_b": 0.02259, "omega_cdm": 0.1238, "H0": 66.11,
        "ln10^{10}A_s": 3.053, "n_s": 0.9666, "tau_reio": 0.0562,
        "A_act": 1.0, "P_act": 1.0,
        "thomson_rescaling": {"prior": {"min": 0.5, "max": 1.5}},              # A_d
        "phenomenological_damping_tail": {"prior": {"min": 0., "max": 1e6}},  # l_d (0 = off)
    },
}

model = get_model(info)
act = model.likelihood[act_name]

################################################################################ Likelihood

def split_chi2():
    """chi2 of TT, TE and EE separately, from the ACT data just evaluated.
    Each uses only that spectrum's data points and its block of the covariance."""
    cl = model.provider.get_Cl(ell_factor=True)
    out = {}
    for pol in ["tt", "te", "ee"]:
        idx, pred = [], []
        for m in act.spec_meta:
            if m["pol"] == pol:
                prediction = m["window"].weight.T @ cl[pol][m["window"].values]
                idx.extend(m["idx"])
                pred.extend(prediction)
        idx = np.array(idx)
        delta = act.data_vec[idx] - np.array(pred)
        out[pol] = delta @ np.linalg.inv(act.covmat[np.ix_(idx, idx)]) @ delta
    return out

def chi2_at(A_d, ell_d):
    """Run CLASS + ACT at one point; return total chi2 and [TT, TE, EE] chi2."""
    lp = model.logposterior({"thomson_rescaling": A_d,
                             "phenomenological_damping_tail": ell_d})
    total = -2 * lp.loglikes[0]
    s = split_chi2()
    return total, [s["tt"], s["te"], s["ee"]]

header = "value chi2_total TT TE EE"

################################################################################ Running 1D scans

A_d_grid = np.unique(np.concatenate((
    np.linspace(0.98, 1.02, 41),        # step 0.001
    [0.8, 0.9, 0.95, 1.05, 1.1, 1.2],   # a few far points for context
)))
rows = []
for A in A_d_grid:
    tot, per = chi2_at(A, 0.)
    rows.append([A, tot, *per])
    print(f"A_d = {A:.4f}   chi2 = {tot:.3f}")
np.savetxt("ACT_scan_A_d.txt", rows, header=header)

ell_d_grid = np.unique(np.concatenate((
    [0.],                                            # LCDM (envelope off)
    np.logspace(np.log10(2e5), np.log10(8e3), 25),   # dense: ACT reaches higher ell than Planck
    [5e5, 1e6],                                      # far out: should converge to LCDM
    [2e3, 3e3, 5e3],                                 # far in: strongly excluded
)))
rows = []
for ld in ell_d_grid:
    tot, per = chi2_at(1.0, ld)
    rows.append([ld, tot, *per])
    print(f"ell_d = {ld:7.0f}   chi2 = {tot:.3f}")
np.savetxt("ACT_scan_ell_d.txt", rows, header=header)
