import numpy
import sharedmem
from common import CovConfig, BootstrapDB
from lib.chunkmap import chunkmap
import mpicov

def main(config):
    DB = BootstrapDB(config)
    Ndof = len(DB.dummy.compress())

    if mpicov.world.rank == 0:
        numpy.random.seed(9999)
        seeds = numpy.random.randint(low=0, high=99999999999, size=config.BigN)
    else:
        seeds = []

    myseeds = mpicov.world.scatter(numpy.array_split(seeds, mpicov.world.size))

    print 'This Task = ', mpicov.world.rank, 'Number of samples = ', \
            len(myseeds), 'seed0 =', myseeds[0]
    myxi = sharedmem.empty((len(myseeds), Ndof), dtype='f8')

    def work(i):
        rng = numpy.random.RandomState(myseeds[i])
        choice = rng.choice(len(DB), size=DB.Nchunks)
        sample = DB(choice)
        myxi[i][...] = sample.compress()

    print 'build samples'
    chunkmap(work, range(len(myxi)), 100)
    print 'done samples'

    print 'covariance matrix'
    cov = mpicov.cov(myxi, rowvar=0, ddof=0)
    print 'done covariance matrix'

    print numpy.nanmin(numpy.diag(cov))
    if mpicov.world.rank == 0:
        numpy.savez(config.CovarianceMatrixOutput, cov=cov,
                BigN=config.BigN, dummy=DB.dummy,
                xi_cov=DB.dummy.copy().uncompress(myxi[0]), r=DB.r, mu=DB.mu)
        
if __name__ == '__main__':
    from sys import argv
    main(CovConfig(argv[1]))
