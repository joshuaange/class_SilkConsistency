import numpy as np
from cobaya.likelihood import Likelihood


class MockCMB(Likelihood):
    """Gaussian likelihood for the mock TT/EE/TE spectra made by make_mock.py."""
    mock_file: str = "mock_Neff.npz"

    def initialize(self):
        d = np.load(self.mock_file)
        self.ell = d["ell"]          # multipoles used
        self.data = d["data"]        # (n_ell, 3): mock TT, EE, TE in muK^2
        self.icov = d["icov"]        # (n_ell, 3, 3): inverse covariance at each ell

    def get_requirements(self):
        lmax = int(self.ell.max())
        return {"Cl": {"tt": lmax, "ee": lmax, "te": lmax}}

    def logp(self, **params):
        cl = self.provider.get_Cl(ell_factor=False, units="muK2")   # lensed C_ell
        theory = np.array([cl["tt"][self.ell], cl["ee"][self.ell], cl["te"][self.ell]]).T
        r = theory - self.data
        return -0.5 * np.einsum("ij,ijk,ik->", r, self.icov, r)
