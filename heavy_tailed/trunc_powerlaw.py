from .base_distribution import distribution
from scipy.optimize import minimize
import numpy as np
from mpmath import mp


class truncated_powerlaw(distribution):
    '''
    truncated shifted power law distribution

    P(x) ~ (x - delta)^{-alpha}, x < x_max,
           zeta(alpha, m_max -delta), x = x_max
    '''
    def __init__(self):
        super(truncated_powerlaw, self).__init__()
        self.name = 'truncated power law'
        self.n_para = 2

    def _loglikelihood(self, alpha_delta, xmin, freq, N, Nmax):
        alpha, delta = alpha_delta
        delta = delta * xmin
        sumlog = np.sum(freq[:-1, -1] * np.log(freq[:-1, 0] - delta))
        logll = Nmax * np.log(float(mp.zeta(alpha, self.xmax - delta)))
        logll -= alpha * sumlog + N * np.log(float(mp.zeta(alpha, xmin - delta)))
        return -logll

    def _fitting(self, xmin=1):
        freq = self.freq[self.freq[:, 0] >= xmin]
        N, Nmax = np.sum(freq[:, -1]), float(freq[-1, -1])
        if xmin not in self.N_xmin:
            self.N_xmin[xmin] = N

        res = minimize(self._loglikelihood, x0=(2, 0.5),
                       method='L-BFGS-B', tol=1e-8,
                       args=(xmin, freq, N, Nmax),
                       bounds=((1. + 1e-2, 5.0), (1e-3, 0.95 - 1e-3)))

        aic = 2 * res.fun + 2 * self.n_para
        fits = {}
        fits['alpha'] = res.x[0]
        fits['delta'] = res.x[1] * xmin
        return (res.x, -res.fun, aic), fits

    def _get_ccdf(self, xmin):
        alpha = self.fitting_res[xmin][1]['alpha']
        delta = self.fitting_res[xmin][1]['delta']
        total, ccdf = 1., []
        normfactor = 1. / float(mp.zeta(alpha, float(xmin - delta)))

        for x in range(xmin, self.xmax):
            total -= (x - delta)**(-alpha) * normfactor
            ccdf.append([x, total])
        return np.asarray(ccdf)
