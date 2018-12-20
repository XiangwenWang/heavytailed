from .lognormal import lognormal
import numpy as np
from scipy.stats import norm


class truncated_lognormal(lognormal):

    def __init__(self):
        super(truncated_lognormal, self).__init__()
        self.name = 'truncated log-normal'
        self.n_para = 3

    def _loglikelihood(self, mu_sigma, freq, xmin, N):
        # P(k)=(Phi((log(k+1)-mu)/sigma)-Phi((log(k)-mu)/sigma)) /
        #       (1-Phi((log(k_min)-mu)/sigma))
        mu, sigma = mu_sigma
        temp_z1 = (np.log(xmin) - mu) / sigma
        temp_z2 = (np.log(self.xmax) - mu) / sigma
        normfactor = self._check_zero_log(norm.cdf(temp_z2) - norm.cdf(temp_z1),
                                          temp_z1)

        lognormfactor = np.array(list(self._norm_factor_i(i, mu, sigma)
                                 for i in freq[:, 0]))
        lognormsum = np.sum(lognormfactor * freq[:, -1])
        logll = lognormsum - N * normfactor
        # print(mu, sigma, -logll)
        return -logll

    def _get_ccdf(self, xmin):
        # P(k)=(Phi((log(k+1)-mu)/sigma)-Phi((log(k)-mu)/sigma))
        #       /(1-Phi((log(k_min)-mu)/sigma))
        mu = self.fitting_res[xmin][1]['mu']
        sigma = self.fitting_res[xmin][1]['sigma']

        total, ccdf = 1., []
        temp_z1 = (np.log(xmin) - mu) / sigma
        temp_z2 = (np.log(self.xmax) - mu) / sigma
        normfactor = self._check_zero_log(norm.cdf(temp_z2) - norm.cdf(temp_z1),
                                          temp_z1)

        for x in range(xmin, self.xmax):
            total -= np.exp(self._norm_factor_i(x, mu, sigma) - normfactor)
            ccdf.append([x, total])

        return np.asarray(ccdf)
