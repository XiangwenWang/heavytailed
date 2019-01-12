from .base_distribution import distribution
import numpy as np
from mpmath import mp
from scipy.optimize import minimize


class powerlaw(distribution):
    '''
    Discrete power law distributions, given by
    P(x) ~ x^(-alpha)
    '''

    def __init__(self):
        super(powerlaw, self).__init__()
        self.name = 'power law'
        self.n_para = 1

    def _loglikelihood(self, alpha_, xmin, logsum_N):
        alpha, = alpha_
        logsum, N = logsum_N
        logll = - alpha * logsum - N * np.log(float(mp.zeta(alpha, xmin)))
        return -logll

    def _fitting(self, xmin=1):
        freq = self.freq[self.freq[:, 0] >= xmin]
        sumlog, N = np.sum(freq[:, -1] * np.log(freq[:, 0])), np.sum(freq[:, -1])
        if xmin not in self.N_xmin:
            self.N_xmin[xmin] = N

        res = minimize(self._loglikelihood, x0=(2.5),
                       method='L-BFGS-B', tol=1e-15,
                       args=(xmin, (sumlog, N)),
                       bounds=((1. + 1e-6, 5.0),))
        aic = 2 * res.fun + 2 * self.n_para
        fits = {}
        fits['alpha'] = res.x[0]
        return (res.x[0], -res.fun, aic), fits

    def _get_ccdf(self, xmin):
        alpha = self.fitting_res[xmin][1]['alpha']
        total, ccdf = 1., []
        normfactor = 1. / float(mp.zeta(alpha, xmin))

        for x in range(xmin, self.xmax):
            total -= x**(-alpha) * normfactor
            ccdf.append([x, total])
        return np.asarray(ccdf)
