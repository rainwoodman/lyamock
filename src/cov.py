# do the covariance matrix.
# this is very hackish it reads mocks from a directory.
#

import numpy
import sharedmem
from kdcount import correlate
import argparse
import os.path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="where to look for mocks and where to put the output")
    A = parser.parse_args() 
    BigN = 10000
    Nchunks = 192

    ids = range(0, 155)

    r, mu = correlate.RmuBinning(80000, Nbins=20, Nmubins=48, observer=0).centers

    fields = ('DQDQ', 'RQDQ', 'RQRQ',
               'DQDFsum1', 'DQDFsum2',
               'RQDFsum1', 'RQDFsum2',
               'DFDFsum1', 'DFDFsum2')
    dtype = numpy.dtype( [
            ( 'DQDQ', ('f8', (len(r), len(mu), Nchunks))),
            ( 'RQDQ', ('f8', (len(r), len(mu), Nchunks))),
            ( 'RQRQ', ('f8', (len(r), len(mu), Nchunks))),
            ( 'DQDFsum1', ('f8', (3, len(r), len(mu), Nchunks))),
            ( 'DQDFsum2', ('f8', (3, len(r), len(mu), Nchunks))),
            ( 'RQDFsum1', ('f8', (3, len(r), len(mu), Nchunks))),
            ( 'RQDFsum2', ('f8', (3, len(r), len(mu), Nchunks))),
            ( 'DFDFsum1', ('f8', (3, len(r), len(mu), Nchunks))),
            ( 'DFDFsum2', ('f8', (3, len(r), len(mu), Nchunks))),
            ]
            )
    sums = sharedmem.empty(len(ids), dtype)
    sizes = sharedmem.empty(len(ids), dtype=[('Q', ('f8', Nchunks)), ('R', ('f8', Nchunks))])

    with sharedmem.MapReduce() as pool:
        def readcorr(i):
            id = ids[i]
            file = numpy.load(os.path.join(A.prefix, '%03d' % id, 'corrbootstrap.npz') )
            for f in fields:
                sums[i][f][...] = file[f][..., 1:-1, 1:-1, :]
            sizes[i]['Q'][...] = file['Qchunksize']
            sizes[i]['R'][...] = file['Rchunksize']

        pool.map(readcorr, range(len(ids)))

    xiQQ = sharedmem.empty((BigN, r.size, mu.size))
    xiFF = sharedmem.empty((BigN, 3, r.size, mu.size ))
    xiQF = sharedmem.empty((BigN, 3, r.size, mu.size))

    seeds = numpy.random.randint(low=0, high=99999999999, size=BigN)
    with sharedmem.MapReduce() as pool:
        def work(i):
            rng = numpy.random.RandomState(seeds[i])
            choice = rng.choice(len(ids), size=Nchunks)
            xiQQ[i, ...] = 0
            sample = numpy.empty((), dtype=dtype)
            Nq = 0
            Nr = 0
            print choice
            for j in range(Nchunks):
                c = choice[j]
                for field in dtype.fields:
                    sample[field][..., j] = sums[field][c, ..., j]
                Nq += sizes['Q'][c, j]
                Nr += sizes['R'][c, j]
            ratio = Nq / Nr
            xiQQ[i] = (sample['DQDQ'].sum(axis=-1) \
                    - sample['RQDQ'].sum(axis=-1) * 2 * ratio) \
                    / (sample['RQRQ'].sum(axis=-1) * ratio ** 2) + 1

            xiQF[i] = (sample['DQDFsum1'].sum(axis=-1) \
                    - sample['RQDFsum1'].sum(axis=-1) * ratio) \
                    / (sample['RQDFsum2'].sum(axis=-1) * ratio)
            xiFF[i] = sample['DFDFsum1'].sum(axis=-1) /sample['DFDFsum2'].sum(axis=-1)
            print 'sample', i, 'Nq', Nq, 'Nr', Nr
        pool.map(work, range(BigN))


    #now lets assemble a giant thing for RSD stuff

    xi_cov = numpy.empty((BigN, 3, r.size, mu.size))

    # 0 is QQ, 1 is QF, 2 is FF
    xi_cov[:, 0, ...] = xiQQ
    xi_cov[:, 1, ...] = xiQF[:, 1, ...]
    xi_cov[:, 2, ...] = xiFF[:, 1, ...]

    cov = numpy.cov(xi_cov.reshape(BigN, -1), rowvar=0)
    cov = cov.reshape(3, r.size, mu.size, 3, r.size, mu.size)
    numpy.savez(os.path.join(A.prefix, 'cov.npz'), cov=cov, BigN=BigN, r=r, mu=mu,
            xi_cov = xi_cov
            xiQQ=xiQQ.mean(axis=0),
            xiQF=xiQF.mean(axis=0),
            xiFF=xiFF.mean(axis=0),
            )

if __name__ == "__main__":
    main()
