#make  the correlation bootstrap.
import numpy
from kdcount import correlate
import chealpy
import sharedmem
from common import Config
from common import CorrFunc, CorrFuncCollection

import os.path

from qsocorr import getqso, getrandom
from pixelcorr import getforest

class RmuBinningIgnoreSelf(correlate.RmuBinning):
    def __call__(self, r, i, j, data1, data2):
        r1 = data1.pos[i]
        r2 = data2.pos[j]
        center = 0.5 * (r1 + r2) - self.observer
        dr = r1 - r2
        dot = numpy.einsum('ij, ij->i', dr, center) 
        center = numpy.einsum('ij, ij->i', center, center) ** 0.5
        div = center * r
        mask = div == 0
        div[mask] = 1.0
        mu = dot / div
        mu[mask] = 10.0
        objid1 = data1.extra[i]
        objid2 = data2.extra[j]
        mu[objid1 == objid2] = 10.0
        return self.linear(r, mu)

def chop(A, Nside, pos):
    """ bootstrap the sky, returns about 100 chunks, only 50 of them are big"""
    # we paint quasar uniformly as long as it is covered by sdss:
    Npix = chealpy.nside2npix(Nside)
    chunkid = sharedmem.empty(len(pos), dtype='intp')
    with sharedmem.MapReduce() as pool:
        chunksize = 1024 * 1024
        def work(i):
            sl = slice(i, i + chunksize)
            chunkid[sl] = chealpy.vec2pix_nest(Nside, pos[sl])
        pool.map(work, range(0, len(pos), chunksize))
    arg = sharedmem.argsort(chunkid)
    chunksize = sharedmem.array.bincount(chunkid, minlength=Npix)
    assert (chunksize == numpy.bincount(chunkid, minlength=Npix)).all()
    return sharedmem.array.packarray(arg, chunksize)

def main(A):
    binning = RmuBinningIgnoreSelf(80000, Nbins=20, Nmubins=48, 
            observer=0)
    r, mu = binning.centers

    qpos = getqso(A)
    rpos = getrandom(A)
    fdelta, fpos, objectid = getforest(A, Zmin=2.0, Zmax=2.2, RfLamMin=1040,
            RfLamMax=1185, combine=12)

    qchunks = chop(A, 4, qpos) 
    rchunks = chop(A, 4, rpos)
    fchunks = chop(A, 4, fpos)
    Nchunks = len(fchunks)
    
    Qfull = correlate.points(qpos, extra=numpy.arange(len(qpos)))
    print 'Qfull:', Qfull.tree.min, Qfull.tree.max
    Rfull = correlate.points(rpos, extra=numpy.arange(len(rpos)))
    print 'Rfull:', Rfull.tree.min, Rfull.tree.max
    Ffull = correlate.field(fpos, value=fdelta, extra=objectid)
    print 'Ffull:', Ffull.tree.min, Ffull.tree.max

    Nvars = fdelta.shape[-1]
    print 'Nchunks', Nchunks, 'Num of variables in fdelta', Nvars
    chunkshape = [Nchunks, binning.shape[0], binning.shape[1]]

    # last index is the chunk
    DQDQ, RQDQ, RQRQ = sharedmem.empty([3] + chunkshape)

    DQDFsum1, RQDFsum1, DFDFsum1 = sharedmem.empty([3, Nvars] + chunkshape)
    DQDFsum2, RQDFsum2, DFDFsum2 = sharedmem.empty([3] + chunkshape)

    with sharedmem.MapReduce() as pool:
        def work(i):
            with pool.critical:
                print 'doing chunk', i, Nchunks

            Qchunk = correlate.points(qpos[qchunks[i]], 
                              extra=Qfull.extra[qchunks[i]])
            Rchunk = correlate.points(rpos[rchunks[i]],
                              extra=Rfull.extra[rchunks[i]])
            Fchunk = correlate.field(fpos[fchunks[i]], 
                    value=fdelta[fchunks[i]],
                    extra=objectid[fchunks[i]] 
                    )
           #Q-Q
            DQDQ[i, ...] = correlate.paircount(Qchunk, Qfull, binning, np=0).fullsum1
            RQDQ[i, ...] = correlate.paircount(Rchunk, Qfull, binning, np=0).fullsum1
            RQRQ[i, ...] = correlate.paircount(Rchunk, Rfull, binning, np=0).fullsum1
           #Q-F
            DQDF = correlate.paircount(Qchunk, Ffull, binning, np=0)
            DQDFsum1[:, i, ...] = DQDF.fullsum1
            DQDFsum2[i, ...] = DQDF.fullsum2
            RQDF = correlate.paircount(Rchunk, Ffull, binning, np=0)
            RQDFsum1[:, i, ...] = RQDF.fullsum1
            RQDFsum2[i, ...] = RQDF.fullsum2
           #F-F
            DFDF = correlate.paircount(Fchunk, Ffull, binning, np=0)
            DFDFsum1[:, i, ...] = DFDF.fullsum1
            DFDFsum2[i, ...] = DFDF.fullsum2
            with pool.critical:
                print 'done chunk', i, Nchunks, len(fchunks[i])
        pool.map(work, range(Nchunks))

    red = MakeBootstrapSample(r, mu, DQDQ, RQDQ, RQRQ,
            DQDFsum1[0], DQDFsum2, RQDFsum1[0], RQDFsum2,
            DFDFsum1[0], DFDFsum2, len(Qfull), len(Rfull))
    real = MakeBootstrapSample(r, mu, DQDQ, RQDQ, RQRQ,
            DQDFsum1[1], DQDFsum2, RQDFsum1[1], RQDFsum2,
            DFDFsum1[1], DFDFsum2, len(Qfull), len(Rfull))
    delta = MakeBootstrapSample(r, mu, DQDQ, RQDQ, RQRQ,
            DQDFsum1[2], DQDFsum2, RQDFsum1[2], RQDFsum2,
            DFDFsum1[2], DFDFsum2, len(Qfull), len(Rfull))

    numpy.savez(os.path.join(A.datadir, 'bootstrap.npz'), 
            r=r,
            mu=mu,
            DQDQ=DQDQ,
            RQDQ=RQDQ,
            RQRQ=RQRQ,
            DQDFsum1=DQDFsum1,
            DQDFsum2=DQDFsum2,
            RQDFsum1=RQDFsum1,
            RQDFsum2=RQDFsum2,
            DFDFsum1=DFDFsum1,
            DFDFsum2=DFDFsum2,
            Qchunksize=qchunks.end - qchunks.start,
            Rchunksize=rchunks.end - rchunks.start,
            Fchunksize=fchunks.end - fchunks.start,
            red=red,
            real=real,
            delta=delta,
            )

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


if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
