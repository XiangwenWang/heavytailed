from .base_distribution import distribution
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


class lognormal(distribution):
    '''
    Discrete log-normal distributions, given by
    ln(x) ~ Normal(mu, sigma^2)

    More specificly:
    P(k)=(Phi((log(k+1)-mu)/sigma)-Phi((log(k)-mu)/sigma))
         / (1-Phi((log(k_min)-mu)/sigma))
    '''

    # change these values when necessary
    # especially when the returned likelihood is np.nan
    para_limits = {'mu': (-100, 100), 'sigma': (1 + 1e-2, 100.)}
    init_values = (0., 2.)

    def __init__(self):
        super(lognormal, self).__init__()
        self.name = 'log-normal'
        self.n_para = 2

    def _norm_factor_i(self, i, mu, sigma):
        normcdf_numerator = (norm.cdf((np.log(i + 1) - mu) / sigma
                                      ) - norm.cdf((np.log(i) - mu) / sigma))
        if normcdf_numerator > 0:
            normcdf = np.log(normcdf_numerator)
        else:
            normcdf = -np.log(2 * np.pi) / 2 - \
                ((np.log(i) - mu) / sigma)**2 / 2
            normcdf += np.log((np.log(i + 1) - np.log(i)) / sigma)
        return normcdf

    def _check_zero_log(self, normfactor, temp_z):
        if normfactor == 0:
            return np.log(1 - np.e**(1.40007 * temp_z)
                          ) - np.log(-temp_z) - temp_z**2 / 2 - 1.04557
        else:
            return np.log(normfactor)

    def _loglikelihood(self, mu_sigma, freq, xmin, N):
        mu, sigma = mu_sigma
        temp_z = -(np.log(xmin) - mu) / sigma
        normfactor = self._check_zero_log(norm.cdf(temp_z), temp_z)

        lognormfactor = np.array(list(self._norm_factor_i(i, mu, sigma)
                                      for i in freq[:, 0]))
        lognormsum = np.sum(lognormfactor * freq[:, -1])
        logll = lognormsum - N * normfactor
        return -logll

    def _fitting(self, xmin=1):
        freq = self.freq[self.freq[:, 0] >= xmin]
        N = np.sum(freq[:, -1])
        if xmin not in self.N_xmin:
            self.N_xmin[xmin] = N

        res2 = minimize(self._loglikelihood, x0=self.init_values,
                        args=(freq, xmin, N),
                        method='L-BFGS-B', tol=1e-8,
                        bounds=(self.para_limits['mu'],
                                self.para_limits['sigma']))
        aic = 2 * res2.fun + 2 * self.n_para
        fits = {}
        fits['mu'] = res2.x[0]
        fits['sigma'] = res2.x[1]
        return (res2.x, -res2.fun, aic), fits

    def _get_ccdf(self, xmin):
        mu = self.fitting_res[xmin][1]['mu']
        sigma = self.fitting_res[xmin][1]['sigma']

        total, ccdf = 1., []
        temp_z = -(np.log(xmin) - mu) / sigma
        normfactor = self._check_zero_log(norm.cdf(temp_z), temp_z)

        for x in range(xmin, self.xmax):
            total -= np.exp(self._norm_factor_i(x, mu, sigma) - normfactor)
            ccdf.append([x, total])

        return np.asarray(ccdf)
