"""ACT DR6 CMB-only likelihood (ACT-lite) restricted to a subset of spectra, e.g. TT only.

Select spectra with the usual `ell_cuts` option, e.g. ell_cuts: {TT: [600, 6500]}.
Spectra not listed in ell_cuts are removed.

Works around a bug in act_dr6_cmbonly.ACTDR6CMBonly: when trailing spectra (TE/EE) are
cut, data_vec ends up shorter than the covariance matrix and the chi^2 fails. Here
data_vec is padded back to full length. The removed entries have zero residual and
(from ACT's own code) a variance of 1e10 with no correlations, so they contribute nothing.
"""
import numpy as np
from act_dr6_cmbonly import ACTDR6CMBonly


class ACTDR6CMBonlySplit(ACTDR6CMBonly):
    def initialize(self):
        super().initialize()
        n = self.covmat.shape[0]
        if len(self.data_vec) < n:
            self.data_vec = np.concatenate([self.data_vec, np.zeros(n - len(self.data_vec))])
