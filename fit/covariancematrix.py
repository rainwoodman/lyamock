import numpy
import sharedmem
from kdcount import correlate
from common import ConfigBase, Config
from bootstrap import MakeBootstrapSample
from lib.chunkmap import chunkmap
import argparse
import os.path

from glob import glob

class BootstrapDB(object):
    def __init__(self, config):
        files = [
                numpy.load(f) 
                for f in glob(os.path.join(config.prefix, '*', '*', 'bootstrap.npz'))]

        self.r = files[0]['r']
        self.mu = files[0]['mu']
        self.DQDQ, self.RQDQ, self.RQRQ = numpy.empty(
                [3, len(files)] + list(files[0]['DQDQ'].shape))
        self.DQDFsum1, self.RQDFsum1, self.DFDFsum1 = numpy.empty(
                [3, len(files)] + list(files[0]['DQDQ'].shape))
        self.DQDFsum2, self.RQDFsum2, self.DFDFsum2 = numpy.empty(
                [3, len(files)] + list(files[0]['DQDQ'].shape))

        self.ND, self.NR = numpy.empty([2, len(files)] +
                list(files[0]['Qchunksize'].shape))

        for i, file in enumerate(files):
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

        self.Nchunks = self.DQDQ[0].shape[0]
        self.dummy = self.buildsample(numpy.zeros(self.Nchunks, 'intp'))

    def __len__(self):
        return len(self.DQDQ)

    def buildsample(self, choice):
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

class CovConfig(ConfigBase):
    def __init__(self, paramfile):
        ConfigBase.__init__(self, paramfile)
        export = self.export
        export("general", "prefix")
        export("general", "BigN", type=int)

def main(config):
    DB = BootstrapDB(config)

    Ndof = len(DB.dummy.compress())

    xi_cov = sharedmem.empty((config.BigN, Ndof), dtype='f8')

    seeds = numpy.random.randint(low=0, high=99999999999, size=config.BigN)

    def work(i):
        rng = numpy.random.RandomState(seeds[i])
        choice = rng.choice(len(DB), size=DB.Nchunks)
        sample = DB.buildsample(choice)
        xi_cov[i][...] = sample.compress()
    chunkmap(work, range(len(xi_cov)), 100)
    print xi_cov  
    cov = parallel_cov(xi_cov, rowvar=0)
    numpy.savez(os.path.join(config.prefix, 'cov.npz'), cov=cov,
            BigN=config.BigN, dummy=DB.dummy, xi_cov=DB.dummy.copy().uncompress(xi_cov[0]), r=DB.r, mu=DB.mu)
    
def parallel_cov(arr, rowvar=1, ddof=None):
    if not rowvar: arr = arr.T
    Nvar = arr.shape[0]
    if ddof is None:
        ddof = 1
    Nsample = arr.shape[1]
    # lets assume this takes little time
    mean = sharedmem.empty(Nvar, dtype='f8')
    with sharedmem.MapReduce() as pool:
        def work(i):
            mean[i] = arr[i].mean(axis=-1)
        pool.map(work, range(Nvar))

    result = sharedmem.empty((Nvar, Nvar), dtype='f8')

    with sharedmem.MapReduce() as pool:
        def work(i):
            di = arr[i] - mean[i]
            co = [1.0 / (Nsample - ddof) * 
                    numpy.sum(di * (arr[j] - mean[j])) \
                    for j in range(Nvar)]
            result[i] = co
            return i
        def reduce(i):
            pass
        pool.map(work, range(Nvar), reduce=reduce)
    return result

if __name__ == '__main__':
    from sys import argv
    main(CovConfig(argv[1]))
