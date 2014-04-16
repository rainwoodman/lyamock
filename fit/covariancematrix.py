import numpy
import sharedmem
from common import CovConfig, BootstrapDB
from lib.chunkmap import chunkmap

def main(config):
    DB = BootstrapDB(config)

    Ndof = len(DB.dummy.compress())

    xi_cov = sharedmem.empty((config.BigN, Ndof), dtype='f8')

    seeds = numpy.random.randint(low=0, high=99999999999, size=config.BigN)

    def work(i):
        rng = numpy.random.RandomState(seeds[i])
        choice = rng.choice(len(DB), size=DB.Nchunks)
        sample = DB(choice)
        xi_cov[i][...] = sample.compress()
    chunkmap(work, range(len(xi_cov)), 100)
    print xi_cov  
    cov = parallel_cov(xi_cov, rowvar=0)
    numpy.savez(config.CovarianceMatrixOutput, cov=cov,
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
