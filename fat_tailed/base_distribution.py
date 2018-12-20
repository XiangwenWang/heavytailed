import numpy as np
import pandas as pd
from scipy.special import gammaln


class distribution(object):
    '''
    This class is used as a skeleton on which we build different
    distribution classes

    When building a specific distribution, one need to inherite this class,
    and re-write functions including:
        __init__() : define the distribution name and number of parameters
        _loglikelihood() : calculate the log-likelihood for a specific
                           parameter setting
        _fitting() : find the maximum likelihood estimator of parameters
        _get_ccdf() : obtain the CCDF

    The functions of this class will
        perform parameter estimation based on maximum likelihood estimation;
        obtain the complementary cumulative distribution function (CCDF),
            also referred as survival function,
            with logarithmic binning;
        calculate the K-S statistics (and perform K-S test);
        and find the start of tails (x_min)
    for a given dataset of discrete values
    '''
    def __init__(self):
        self.name = 'distribution not specified'
        self.fitting_res = {}
        self.ccdf_x = {}
        self.ccdf_data = {}
        self.ccdf = {}
        self.N_xmin = {}
        self.KSStat = {}
        self.KSTest = {}
        self._KSStat_RR = {0.05: 1.35810, 0.1: 1.22385}

    def _loglikelihood(self):
        # will be overrided in specific distribution class
        pass

    def _fitting(self):
        # will be overrided in specific distribution class
        pass

    def _get_ccdf(self):
        # will be overrided in specific distribution class
        pass

    def prepare(self, filename):
        self.filename = filename
        self._load_data()
        self.xmax = max(self.freq[:, 0])

    def _value_counts(self, data):
        total, freq = len(data), []

        for x in np.sort(np.unique(data))[:]:
            data = data[data >= x]
            count_x = len(data[data == x])
            total -= count_x
            freq.append([x, total, len(data), count_x])

        self.freq = np.asarray(freq, dtype=int)

    def _load_data(self):
        try:
            self.freq_filename = self.filename.replace('raw_', 'freq_')
            self.freq = np.loadtxt(self.freq_filename, dtype=int)
        except (IOError, ValueError):
            data = pd.read_csv(self.filename, dtype=int,
                               header=None, names=['data']).data.values
            self._value_counts(data)
            np.savetxt(self.freq_filename, self.freq, fmt='%d')

    def _sum_log_gamma_func(self, freq, shift=0):
        return np.sum(gammaln(freq[:, 0] + shift) * freq[:, -1])

    def _round_res(self, res):
        para, ll, aic = res
        if type(para) == np.float64:
            para = str(round(para, 8))
        else:
            para = ' '.join(map(lambda x: str(round(x, 8)), para))
        ll = str(int(round(ll, 0)))
        aic = str(int(round(aic, 0)))
        # return '\t'.join([para, ll, aic])
        return '\t'.join([para, aic])

    def fit(self, filename=None, xmin=1, output=True):
        try:
            if self.filename != filename:
                if filename is not None:
                    self.prepare(filename)
                    print('Filename changed')
        except AttributeError:
            self.prepare(filename)

        self.fitting_res[xmin] = self._fitting(xmin=xmin)
        if output:
            print('xmin=%d\t' % xmin + self.name.title() + '\t' +
                  self._round_res(self.fitting_res[xmin][0]))

        return self.fitting_res[xmin][1]

    def _get_data_ccdf(self, xmin, save_filename, long_tail=True):

        freq = self.freq[self.freq[:, 0] >= xmin]
        N = freq[0, 2]

        if not long_tail:
            # in short-tail cases, return unbinned results

            self.ccdf_data[xmin] = np.asarray([freq[:-1, 0], freq[:-1, 1] / N
                                               ], dtype=float).T
            return

        bins = 0.1
        threshold = .5  # percents
        filt_y = (100 / threshold)**2
        ccdf_bd = []  # binned ccdf
        binned = False

        for i in range(len(freq[:-1])):
            if (not binned) and ((freq[i, -1] > filt_y) or
                                 (freq[i, 0] < 2) or
                                 np.log10(freq[i, 0]) -
                                 np.log10(freq[i, 0] - 1) > bins):
                ccdf_bd.append([freq[i, 0], freq[i, 1]])

            else:
                binned = True
                mid_x, last_mid_index = 0, i - 1
                while mid_x < self.xmax - 1:
                    if not mid_x:
                        start_log_x = np.log10(freq[i][0] - 0.5)
                    else:
                        mid_index = freq[:, 0].searchsorted(mid_x)
                        if last_mid_index != mid_index:
                            ccdf_bd.append([freq[mid_index][0], freq[mid_index][1]])
                        last_mid_index = mid_index
                    try:
                        mid_x = int(np.power(10, start_log_x + bins / 2.)) + 1
                    except:
                        break
                    start_log_x += bins
                break

        ccdf_bd = np.asarray(ccdf_bd, dtype=float)
        ccdf_bd[:, 1] /= N
        if ccdf_bd[-1, -1] < 1e-18:
            ccdf_bd = ccdf_bd[: -1]
        self.ccdf_data[xmin] = ccdf_bd

    def check(self, xmin=1, alpha=0.05, output=True):
        try:
            ccdf_data_filename = self.filename.replace('raw_', ''
                ).replace('.dat', '_ccdf_%d.dat' % xmin)
            self.ccdf_data[xmin] = np.loadtxt(ccdf_data_filename)
        except (IOError, ValueError):
            self._get_data_ccdf(xmin, ccdf_data_filename)
            np.savetxt(ccdf_data_filename, self.ccdf_data[xmin])

        self.ccdf_x[xmin] = np.asarray(self.ccdf_data[xmin][:, 0], dtype=int)

        ccdf = self._get_ccdf(xmin)
        ccdf_bd = []
        for x in self.ccdf_x[xmin]:
            x_ccdf_index = ccdf[:, 0].searchsorted(x)
            if x_ccdf_index == len(ccdf):
                continue
            ccdf_bd.append(ccdf[x_ccdf_index])
        self.ccdf[xmin] = np.asarray(ccdf_bd, dtype=float)
        ccdf_filename = ccdf_data_filename.replace('_ccdf_', '_ccdf_%s_' %
                                                   self.name.replace(' ', '_'))
        np.savetxt(ccdf_filename, self.ccdf[xmin])

        self.KSStat[xmin] = np.max(np.abs(self.ccdf_data[xmin][:, 1] - self.ccdf[xmin][:, 1]))
        if self.KSStat[xmin] > self._KSStat_RR[alpha] / np.sqrt(self.N_xmin[xmin]):
            self.KSTest[xmin] = False
        else:
            self.KSTest[xmin] = True
        if output:
            print('K-S Test: %.6f %s' % (self.KSStat[xmin], self.KSTest[xmin]))

    def find_xmin(self, xmin=None, xmax=None, showall=False, top=1):
        top_res = None
        xmin = xmin if xmin is not None else min(self.KSStat)
        xmax = xmax if xmax is not None else max(self.KSStat)
        print(self.filename, self.name.title())

        if showall:
            for i in sorted(self.KSStat):
                if xmin <= i <= xmax:
                    print(i, '%.8f' % self.KSStat[i], self.KSTest[i])
        else:
            output_count = 0
            for i in sorted(self.KSStat, key=self.KSStat.get):
                if xmin <= i <= xmax:
                    if top_res is None:
                        top_res = i
                    print(i, '%.8f' % self.KSStat[i], self.KSTest[i])
                    output_count += 1
                    if output_count == top:
                        break
        return top_res
