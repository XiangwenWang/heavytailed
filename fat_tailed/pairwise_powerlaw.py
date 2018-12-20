from .base_distribution import distribution
import numpy as np
from mpmath import mp
from scipy.optimize import minimize


class pairwise_powerlaw(distribution):
    '''
    Discrete pairwise power law distribution, given by
    P(x) ~ x^(-alpha)   for xmin <= x < x_trans
           x^(-beta)    for x >= x_trans
    '''
    def __init__(self):
        super(pairwise_powerlaw, self).__init__()
        self.name = 'pairwise power law'
        self.n_para = 3

    def _loglikelihood(self, alpha_beta_dtrans, xmin, freq, N):
        alpha, beta, dtrans0 = alpha_beta_dtrans
        dtrans = xmin + (self.xmax - xmin) * dtrans0
        dtrans_ceil = np.ceil(dtrans)

        logc = -np.log(float(mp.zeta(alpha, xmin)) -
                       float(mp.zeta(alpha, dtrans_ceil)) +
                       dtrans**(beta - alpha) *
                       float(mp.zeta(beta, dtrans_ceil)))

        region1_freq = freq[freq[:, 0] < dtrans_ceil]
        region2_freq = freq[freq[:, 0] >= dtrans_ceil]
        logll = logc * N
        logll -= alpha * np.sum(np.log(region1_freq[:, 0]) *
                                region1_freq[:, -1])
        logll -= beta * np.sum(np.log(region2_freq[:, 0]) *
                               region2_freq[:, -1])
        logll += (beta - alpha) * np.sum(region2_freq[:, -1]
                                         ) * np.log(dtrans)
        return -logll

    def _fitting(self, xmin=1):
        freq = self.freq[self.freq[:, 0] >= xmin]
        N = np.sum(freq[:, -1])
        if xmin not in self.N_xmin:
            self.N_xmin[xmin] = N

        res = minimize(self._loglikelihood, x0=(1.3, 3.5, 0.001),
                       method='L-BFGS-B', tol=1e-8,
                       args=(xmin, freq, N),
                       bounds=((0. + 1e-6, 10.0),
                               (1. + 1e-6, 10.0),
                               (0, 1)))
        aic = 2 * res.fun + 2 * self.n_para
        fits = {}
        fits['alpha'] = res.x[0]
        fits['beta'] = res.x[1]
        fits['dtrans'] = res.x[2] * (self.xmax - xmin) + xmin
        return (res.x, -res.fun, aic), fits

    def _get_ccdf(self, xmin):

        alpha = self.fitting_res[xmin][1]['alpha']
        beta = self.fitting_res[xmin][1]['beta']
        dtrans = self.fitting_res[xmin][1]['dtrans']

        total, ccdf = 1., []
        dtrans_ceil = int(np.ceil(dtrans))
        c = 1. / (float(mp.zeta(alpha, xmin)) -
                  float(mp.zeta(alpha, dtrans_ceil)) +
                  dtrans**(beta - alpha) * float(mp.zeta(beta, dtrans_ceil)))

        for x in range(xmin, dtrans_ceil):
            total -= x**(-alpha) * c
            ccdf.append([x, total])
        for x in range(dtrans_ceil, self.xmax):
            total -= x**(-beta) * c * dtrans**(beta - alpha)
            ccdf.append([x, total])

        return np.asarray(ccdf)
