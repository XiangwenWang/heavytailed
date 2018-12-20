import numpy as np
from json import dump as savejson
import matplotlib.pyplot as plt
from .powerlaw import powerlaw
from .yule import yule
from .trunc_powerlaw import truncated_powerlaw
from .poisson import poisson
from .shifted_powerlaw_exp import shifted_powerlaw_exp
from .pairwise_powerlaw import pairwise_powerlaw
from .exponential import exponential
from .lognormal import lognormal
from .powerlaw_exp import powerlaw_exp
from .trunc_lognormal import truncated_lognormal


class comparison(object):

    valid_dists = {'power law': powerlaw,
                   'exponential': exponential,
                   'log-normal': lognormal,
                   # 'truncated log-normal': truncated_lognormal,
                   'pairwise power law': pairwise_powerlaw,
                   'power law with exp cutoff': powerlaw_exp,
                   'yule': yule,
                   'poisson': poisson,
                   # 'shifted power law with exp cutoff': shifted_powerlaw_exp,
                   # 'truncated power law': truncated_powerlaw,
                   }

    all_dists = {'power law': powerlaw,
                 'exponential': exponential,
                 'log-normal': lognormal,
                 'truncated log-normal': truncated_lognormal,
                 'pairwise power law': pairwise_powerlaw,
                 'power law with exp cutoff': powerlaw_exp,
                 'yule': yule,
                 'poisson': poisson,
                 'shifted power law with exp cutoff': shifted_powerlaw_exp,
                 'truncated power law': truncated_powerlaw,
                 }

    def __init__(self, filename, distributions=None, xmin=None,
                 draw=False, output=True, outputKS=False):

        if distributions is None:
            distributions = self.valid_dists
        self.dists = {}
        self.filename = filename
        self.fitting_res = {}
        self.ccdfs = {}

        for d in distributions:
            if d not in self.valid_dists:
                print('Specified distribution is not valid')
                print('Please choose one from')
                print(self.valid_dists)
            else:
                self.dists[d] = self.valid_dists[d]()

        if xmin is not None:
            self.fit(xmin, draw=draw, output=output, outputKS=outputKS)

    def update_distributions(self, new_dists):
        for d in self.dists:
            if d not in new_dists:
                self.dists.pop(d)
                self.fitting_res.pop(d)

    def fit(self, xmin, draw=False, output=True, outputKS=False):
        if type(xmin) is int:
            self._fit_xmin(xmin, draw=draw, output=output, outputKS=outputKS)
        else:
            try:
                for i in xmin:
                    self._fit_xmin(int(i), draw=draw,
                                   output=output, outputKS=outputKS)
            except TypeError:
                self._fit_xmin(int(xmin), draw=draw,
                               output=output, outputKS=outputKS)

    def _fit_xmin(self, xmin, draw=False, output=True, outputKS=False):
        fitting_res_xmin = {}
        for d in self.dists:
            fitting_res_xmin[d] = self.dists[d].fit(self.filename, xmin=xmin,
                                                    output=output)
            self.dists[d].check(xmin=xmin, output=outputKS)

        self.fitting_res[xmin] = fitting_res_xmin
        fit_res_file = self.filename.replace('raw_', 'fitting_'
                                             ).replace('.dat',
                                                       '_%d.json' % xmin)
        with open(fit_res_file, 'w') as fp:
            savejson(fitting_res_xmin, fp)
            fp.close()

        self._get_all_ccdfs(xmin, draw=draw)

    def _get_all_ccdfs(self, xmin, draw=False):
        self.ccdfs[xmin] = {}
        ccdf_data_filename = self.filename.replace('raw_', ''
                              ).replace('.dat', '_ccdf_%d.dat' % xmin)
        self.ccdfs[xmin]['data'] = np.loadtxt(ccdf_data_filename)

        for d in self.dists:
            ccdf_d_filename = ccdf_data_filename.replace('_ccdf_', '_ccdf_%s_' %
                                                         d.replace(' ', '_'))
            self.ccdfs[xmin][d] = np.loadtxt(ccdf_d_filename)

        if draw:
            self.draw_ccdf(xmin=xmin)

    def draw_ccdf(self, xmin=1, distributions=None):
        if distributions is None:
            distributions = self.dists
        for d in distributions:
            self._draw_ccdf(self.ccdfs[xmin]['data'], self.ccdfs[xmin][d],
                            d, xmin=xmin)

    def _draw_ccdf(self, ccdf_data, ccdf_d, d, xmin=1):
        plt.figure(figsize=(6, 4.5))
        fig1, = plt.plot(ccdf_data[:, 0], ccdf_data[:, 1], 'bo')
        fig2, = plt.plot(ccdf_d[:, 0], ccdf_d[:, 1], 'r-', linewidth=2)
        plt.legend((fig1, fig2), ('data', d))
        plt.title('%s xmin=%d' % (self.filename, xmin))
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
