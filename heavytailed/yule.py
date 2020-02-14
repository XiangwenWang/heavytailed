from .base_distribution import distribution
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize


class yule(distribution):
    '''
    Yule distributions, given by
    P(x) ~ gamma(x) / gamma(x + alpha)
    '''

    def __init__(self):
        super(yule, self).__init__()
        self.name = 'yule'
        self.n_para = 1

    def _loglikelihood(self, alpha_, xmin, freq, sum_log_gamma, N):
        alpha, = alpha_
        logll = N * (np.log(alpha - 1.) + gammaln(xmin + alpha - 1) -
                     gammaln(xmin))
        logll += sum_log_gamma
        logll -= self._sum_log_gamma_func(freq, shift=alpha)
        return -logll

    def _fitting(self, xmin=1):
        freq = self.freq[self.freq[:, 0] >= xmin]
        N = np.sum(freq[:, -1])
        sum_log_gamma = self._sum_log_gamma_func(freq)
        if xmin not in self.N_xmin:
            self.N_xmin[xmin] = N

        res = minimize(self._loglikelihood, x0=(1.6),
                       method='SLSQP', tol=1e-8,
                       args=(xmin, freq, sum_log_gamma, N),
                       bounds=((1. + 1e-6, 100.0),))
        aic = 2 * res.fun + 2 * self.n_para
        fits = {}
        fits['alpha'] = res.x[0]
        return (res.x[0], -res.fun, aic), fits

    def _get_ccdf(self, xmin):
        alpha = self.fitting_res[xmin][1]['alpha']
        total, ccdf = 1., []
        normfactor = (alpha - 1.) * np.exp(gammaln(xmin + alpha - 1
                                                   ) - gammaln(xmin))

        for x in range(xmin, self.xmax):
            total -= np.exp(gammaln(x) - gammaln(x + alpha)) * normfactor
            ccdf.append([x, total])

        return np.asarray(ccdf)
