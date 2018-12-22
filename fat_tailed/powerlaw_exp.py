from .base_distribution import distribution
import numpy as np
from scipy.optimize import minimize
from mpmath import mp


class powerlaw_exp(distribution):
    '''
    Power law distributions with exponential cutoff, given by
    P(x) ~ x^(-alpha) e^(-lambda x)
    '''

    def __init__(self):
        super(powerlaw_exp, self).__init__()
        self.name = 'power law with exp cutoff'
        self.n_para = 2

    def _loglikelihood(self, alpha_lambda, xmin,
                       sumdata, sumlog, N):
        # P(x) ~ x^-alpha e^(-lambda x)
        alpha, lambda_ = alpha_lambda
        norm_factor = float(mp.polylog(alpha, np.exp(-lambda_)))

        xmin_array = np.arange(1, xmin)
        norm_factor -= np.sum(np.power(xmin_array, -alpha) *
                              np.exp(-lambda_ * xmin_array))
        logll = - (alpha * sumlog + lambda_ * sumdata +
                   N * np.log(norm_factor))

        return -logll

    def _fitting(self, xmin=1):
        freq = self.freq[self.freq[:, 0] >= xmin]
        sumdata = np.sum(freq[:, -1] * freq[:, 0])
        sumlog = np.sum(freq[:, -1] * np.log(freq[:, 0]))
        N = np.sum(freq[:, -1])
        if xmin not in self.N_xmin:
            self.N_xmin[xmin] = N

        res = minimize(self._loglikelihood, x0=(2, 1e-4),
                       method='L-BFGS-B', tol=1e-8,
                       args=(xmin, sumdata, sumlog, N),
                       bounds=((1e-15, 5), (1e-15, 1e-1)))

        aic = 2 * res.fun + 2 * self.n_para
        fits = {}
        fits['alpha'] = res.x[0]
        fits['lambda'] = res.x[1]
        return (res.x, -res.fun, aic), fits

    def _get_ccdf(self, xmin):
        # P(k) = 1./Li_alpha(e^(-lambda)) * x^(-alpha) * exp(-lambda x)
        # where Li_s(z) is the polylogarithmic function

        alpha = self.fitting_res[xmin][1]['alpha']
        lambda_ = self.fitting_res[xmin][1]['lambda']

        total, ccdf = 1., []
        norm_denom = float(mp.polylog(alpha, np.exp(-lambda_)))
        xmin_array = np.arange(1, xmin)
        norm_denom -= np.sum(np.power(xmin_array, -alpha) *
                             np.exp(-lambda_ * xmin_array))
        normfactor = 1. / norm_denom

        for x in range(xmin, self.xmax):
            total -= x**(-alpha) * np.exp(-lambda_ * x) * normfactor
            ccdf.append([x, total])

        return np.asarray(ccdf)
