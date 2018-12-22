from .base_distribution import distribution
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson as sp_pois
from scipy.special import gammaln


class poisson(distribution):
    '''
    (Truncated) poisson distributions, given by
    P(x) ~ mu^x / x!

    This is a thin-tailed distribution
    '''
    def __init__(self):
        super(poisson, self).__init__()
        self.name = 'poisson'
        self.n_para = 1

    def _loglikelihood(self, mu_, xmin, sum_x, sum_log_x_fact, N):
        mu, = mu_
        logll = sum_x * np.log(mu) - sum_log_x_fact
        logll -= N * (mu + np.log(1 - sp_pois.cdf(xmin - 1, mu)))
        return -logll

    def _fitting(self, xmin=1):
        freq = self.freq[self.freq[:, 0] >= xmin]
        sum_x = np.sum(freq[:, 0] * freq[:, -1])
        sum_log_x_fact = self._sum_log_gamma_func(freq, shift=1)
        N = np.sum(freq[:, -1])
        if xmin not in self.N_xmin:
            self.N_xmin[xmin] = N

        res = minimize(self._loglikelihood, x0=(xmin + 1.),
                       method='L-BFGS-B', tol=1e-8,
                       args=(xmin, sum_x, sum_log_x_fact, N),
                       bounds=((0.1 + 1e-6, 10.),))

        aic = 2 * res.fun + 2 * self.n_para
        fits = {}
        fits['mu'] = res.x[0]
        return (res.x[0], -res.fun, aic), fits

    def _get_ccdf(self, xmin):
        mu = self.fitting_res[xmin][1]['mu']
        total, ccdf = 1, []
        normfactor = 1. / np.exp(mu + np.log(1 - sp_pois.cdf(xmin - 1, mu)))

        for x in range(xmin, self.xmax):
            total -= np.exp(x * np.log(mu) - gammaln(x + 1)) * normfactor
            ccdf.append([x, total])

        return np.asarray(ccdf)
