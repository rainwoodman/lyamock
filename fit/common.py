import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from lib.common import *
import sharedmem
from lib.chunkmap import chunkmap

def writepickle():
    import pickle
    from sys import stdout
    mu = numpy.linspace(-1, 1, 10)
    r = numpy.linspace(-1, 1, 10)
    xi = numpy.empty((len(r), len(mu)))
    s = pickle.dumps(CorrFuncCollection([CorrFunc(r, mu, xi)]))
    stdout.write(s)

def readpickle():
    import pickle
    from sys import stdin
    print pickle.load(stdin)

import os.path
from glob import glob


class CovConfig(ConfigBase):
    def __init__(self, paramfile):
        ConfigBase.__init__(self, paramfile)
        export = self.export

        export("General", "prefix")
        export("General", "UseMocks", type=eval, default="None")
        export("General", "PowerSpectrumCache")
        export("General", "BigN", type=int)
        export("Fit", ["rmin", "rmax"], type=float)
        export("General", "CovarianceMatrixOutput",
                default=os.path.join(self.prefix, "cov.npz"))
        export("General", "EigenModesOutput",
                default=os.path.join(self.prefix, "eigenmodes.npz"))

        export("Cosmology", [
            "Sigma8",
            "OmegaM",
            "OmegaB",
            "OmegaL", 
            "h"] , type=float)

        self.cosmology = Cosmology(M=self.OmegaM, 
            L=self.OmegaL, B=self.OmegaB, h=self.h, sigma8=self.Sigma8)

class BootstrapDB(object):
    def __init__(self, config):
        def getfilename(mock):
            dir = os.path.join(config.prefix, mock)
            paramfile = os.path.join(config.prefix, mock, 'paramfile')
            c = Config(paramfile, basedir=dir)
            return os.path.join(c.datadir, 'bootstrap.npz')

        if config.UseMocks is None:
            filenames = sorted(list(glob(os.path.join(config.prefix, '[0-9]*', '*',
                'bootstrap.npz'))))
        else:
            filenames = [getfilename(mock) for mock in config.UseMocks]

        files = [ numpy.load(f)  for f in filenames]

        print 'using', len(filenames), ' files', filenames

        self.r = files[0]['r']
        self.mu = files[0]['mu']

        # b/c they all have the same cosmology
        self.eigenmodes = EigenModes(files[0]['eigenmodes'])

        self.DQDQ, self.RQDQ, self.RQRQ = sharedmem.empty(
                [3, len(files)] + list(files[0]['DQDQ'].shape))
        self.DQDFsum1, self.RQDFsum1, self.DFDFsum1 = sharedmem.empty(
                [3, len(files)] + list(files[0]['DQDQ'].shape))
        self.DQDFsum2, self.RQDFsum2, self.DFDFsum2 = sharedmem.empty(
                [3, len(files)] + list(files[0]['DQDQ'].shape))

        self.ND, self.NR = sharedmem.empty([2, len(files)] +
                list(files[0]['Qchunksize'].shape))

        def read(i):
            file = files[i]
            self.DQDQ[i] = file['DQDQ']
            self.RQDQ[i] = file['RQDQ']
            self.RQRQ[i] = file['RQRQ']
            self.DQDFsum1[i] = file['DQDFsum1'][0]
            self.RQDFsum1[i] = file['RQDFsum1'][0]
            self.DFDFsum1[i] = file['DFDFsum1'][0]
            self.DQDFsum2[i] = file['DQDFsum2']
            self.RQDFsum2[i] = file['RQDFsum2']
            self.DFDFsum2[i] = file['DFDFsum2']
            self.ND[i] = file['Qchunksize']
            self.NR[i] = file['Rchunksize']

        chunkmap(read, range(len(files)), 1)

        self.Nchunks = self.DQDQ[0].shape[0]
        # build the currelation function on the first sample
        # use it as a template 
        self.dummy = self(0)

    def __len__(self):
        return len(self.DQDQ)

    def __call__(self, choice):
        if numpy.isscalar(choice):
            choice = numpy.repeat(choice, self.Nchunks)
        def mytake(a, c):
            newshape = [-1] + list(a.shape[2:])
            a = a.reshape(newshape)
            return a.take(c * self.Nchunks + numpy.arange(self.Nchunks), axis=0)
        DQDQ = mytake(self.DQDQ, choice)
        RQDQ = mytake(self.RQDQ, choice)
        RQRQ = mytake(self.RQRQ, choice)
        DQDFsum1 = mytake(self.DQDFsum1, choice)
        DQDFsum2 = mytake(self.DQDFsum2, choice)
        RQDFsum1 = mytake(self.RQDFsum1, choice)
        RQDFsum2 = mytake(self.RQDFsum2, choice)
        DFDFsum1 = mytake(self.DFDFsum1, choice)
        DFDFsum2 = mytake(self.DFDFsum2, choice)
        ND = mytake(self.ND, choice)
        NR = mytake(self.NR, choice)

        return MakeBootstrapSample(self.r, self.mu,
                DQDQ, RQDQ, RQRQ, DQDFsum1, DQDFsum2, RQDFsum1, RQDFsum2,
                DFDFsum1, DFDFsum2, ND.sum(), NR.sum())

def MakeBootstrapSample(r, mu,
        DQDQ, RQDQ, RQRQ,
        DQDFsum1, DQDFsum2, 
        RQDFsum1, RQDFsum2,
        DFDFsum1, DFDFsum2,
        Nq, Nr):
        ratio = 1.0 * Nq / Nr
        return CorrFuncCollection([
            CorrFunc.QQ(r, mu, DQDQ.sum(axis=0), RQDQ.sum(axis=0),
                RQRQ.sum(axis=0), ratio),
            CorrFunc.QF(r, mu, DQDFsum1.sum(axis=0), RQDFsum1.sum(axis=0), 
             RQDFsum2.sum(axis=0), ratio),
            CorrFunc.FF(r, mu, DFDFsum1.sum(axis=0), DFDFsum2.sum(axis=0))])

class EigenModes(list):
    def __init__(self, newlist):
        list.__init__(self, [CorrFuncCollection(e) for e in newlist])
        # for fast construction of a mode.
        self.modes = [
            CorrFuncCollection(e).compress()
            for e in newlist ]
    def __getstate__(self):
        return (list(self),)

    def __setstate__(self, state):
        self.__init__(state[0])

    def __call__(self, p):
        """ bF, bQ, BF, BQ"""
        bF, bQ, BF, BQ=p
        coeff = [ 
                co(bQ, BQ, bF, BF) 
                for co in [
                    C0QQ, C2QQ, C4QQ, 
                    C0QF, C2QF, C4QF, 
                    C0FF, C2FF, C4FF]
                ]
        xi = reduce(numpy.add, 
                [m * c for m, c in zip(self.modes, coeff)])
        return self[0].copy().uncompress(xi)

def C0FF(bQ, BQ, bF, BF):
    return bF ** 2 * (1 + 2. / 3. * BF + 1. / 5. * BF ** 2)
def C2FF(bQ, BQ, bF, BF):
    return -bF ** 2 * (4. / 3. * BF + 4. / 7. * BF ** 2)
def C4FF(bQ, BQ, bF, BF):
    return bF ** 2 * (8. / 35. * BF ** 2)

def C0QQ(bQ, BQ, bF, BF):
    return bQ ** 2 * (1 + 2. / 3. * BQ + 1. / 5. * BQ ** 2)
def C2QQ(bQ, BQ, bF, BF):
    return -bQ ** 2 * (4. / 3. * BQ + 4. / 7. * BQ ** 2)
def C4QQ(bQ, BQ, bF, BF):
    return bQ ** 2 * (8. / 35. * BQ ** 2)

def C0QF(bQ, BQ, bF, BF):
    return bF * bQ * (1 + 1./3 * (BF + BQ) + 1. / 5 * BF * BQ)
def C2QF(bQ, BQ, bF, BF):
    return -bF * bQ *(2./3 * (BF + BQ) + 4. / 7 * BF * BQ)
def C4QF(bQ, BQ, bF, BF):
    return bF * bQ * (8. / 35. * BF * BQ ** 2)


