from .base_distribution import distribution
import numpy as np


class exponential(distribution):
    '''
    Discrete exponential distributions (shifted geometric), given by
    P(x) ~ e^(-x * lambda)
    '''
    def __init__(self):
        super(exponential, self).__init__()
        self.name = 'exponential'
        self.n_para = 1

    def _fitting(self, xmin=1):
        # P(x) ~ e^(-lambda x)
        freq = self.freq[self.freq[:, 0] >= xmin]
        N = np.sum(freq[:, -1])
        if xmin not in self.N_xmin:
            self.N_xmin[xmin] = N
        x_bar = 1. * np.sum(freq[:, 0] * freq[:, -1]) / np.sum(freq[:, -1])
        p = 1. / (x_bar - (xmin - 1))
        mu = - np.log(1 - p)
        logll = N * (x_bar - xmin) * np.log(1 - p) + N * np.log(p)
        aic = - 2 * logll + 2 * self.n_para
        fits = {}
        fits['mu'] = mu
        return (mu, logll, aic), fits

    def _get_ccdf(self, xmin):
        mu = self.fitting_res[xmin][1]['mu']
        total, ccdf = 1., []
        normfactor = (1 - np.exp(-mu)) * np.exp(mu * xmin)

        for x in range(xmin, self.xmax):
            total -= np.exp(-mu * x) * normfactor
            ccdf.append([x, total])
        return np.asarray(ccdf)
