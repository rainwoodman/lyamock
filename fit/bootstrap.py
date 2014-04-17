#make  the correlation bootstrap.
import numpy
from kdcount import correlate
import chealpy
import sharedmem
from common import Config
from common import PowerSpectrum
from common import EigenModes

from common import CorrFunc, CorrFuncCollection
from common import MakeBootstrapSample

from numpy.polynomial.legendre import legval

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
    print len(pos)
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
    binning = RmuBinningIgnoreSelf(160000, Nbins=40, Nmubins=48, 
            observer=0)
    r, mu = binning.centers

    qpos = getqso(A)
    rpos = getrandom(A)
    fdelta, fpos, objectid = getforest(A, Zmin=2.0, Zmax=3.0, RfLamMin=1040,
           RfLamMax=1216, combine=4)
#    fdelta, fpos = numpy.empty((2, 1, 3))
#    objectid = numpy.empty(1, dtype='i8')

    qchunks = chop(A, 4, qpos) 
    rchunks = chop(A, 4, rpos)
    fchunks = chop(A, 4, fpos)
    Nchunks = len(qchunks)
    
    Nvars = fdelta.shape[-1]
    print 'Nchunks', Nchunks, 'Num of variables in fdelta', Nvars
    chunkshape = [Nchunks, binning.shape[0], binning.shape[1]]

    # last index is the chunk
    DQDQ, RQDQ, RQRQ = sharedmem.empty([3] + chunkshape)

    DQDFsum1, RQDFsum1, DFDFsum1 = sharedmem.empty([3, Nvars] + chunkshape)
    DQDFsum2, RQDFsum2, DFDFsum2 = sharedmem.empty([3] + chunkshape)

    with sharedmem.MapReduce() as pool:
        Qfull = correlate.points(qpos, extra=numpy.arange(len(qpos)))
        print 'Qfull:', Qfull.tree.min, Qfull.tree.max
        Rfull = correlate.points(rpos, extra=numpy.arange(len(rpos)))
        print 'Rfull:', Rfull.tree.min, Rfull.tree.max
        Ffull = correlate.field(fpos, value=fdelta, extra=objectid)
        print 'Ffull:', Ffull.tree.min, Ffull.tree.max

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
            DFDFsum1[0], DFDFsum2, len(qpos), len(rpos))
    real = MakeBootstrapSample(r, mu, DQDQ, RQDQ, RQRQ,
            DQDFsum1[1], DQDFsum2, RQDFsum1[1], RQDFsum2,
            DFDFsum1[1], DFDFsum2, len(qpos), len(rpos))
    delta = MakeBootstrapSample(r, mu, DQDQ, RQDQ, RQRQ,
            DQDFsum1[2], DQDFsum2, RQDFsum1[2], RQDFsum2,
            DFDFsum1[2], DFDFsum2, len(qpos), len(rpos))

    powerspec = PowerSpectrum(A)
    eigenmodes = MakeEigenModes(powerspec, red)
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
            eigenmodes=eigenmodes,
            )

def MakeEigenModes(powerspec, templatecorrfunc):
    """ create fitting eigenstates for a given powerspectrum
        see Slosar paper.

        basically getting out the poles of the powerspectrum,
        then evaluate them on the grid given in templatecorrfunc.
        this is used in bootstrap (for per sample)

        usually, the eigenmodes are ordered first by the correlation functions, 
        (QQ QF, FF) then by the multipole order (0, 2, 4)
    """
    dummy = templatecorrfunc

    eigenmodes = []
    N = len(dummy.compress())
    annotation = []

    for i in range(len(dummy)):
        for order in [0, 2, 4]:
            annotation.append((i, order))
            eigenmode = dummy.copy()
            eigenmode.uncompress(numpy.zeros(N))
            eigenmode[i].xi = sharedmem.copy(eigenmode[i].xi)
            eigenmodes.append(eigenmode)

    with sharedmem.MapReduce() as pool:
        def work(j):
            i, order = annotation[j]
            c = numpy.zeros(5)
            c[order] = 1
            # watch out the eigenmode is wikipedia
            # no negative phase on 2nd order pole!
            eigenmode = eigenmodes[j]
            eigenmode[i].xi[...] = \
                    powerspec.pole(eigenmode[i].r, order=order)[:, None] \
                    * legval(eigenmode[i].mu, c)[None, :]
        pool.map(work, range(len(annotation)))
    # the eigenmodes are ordered first by the correlation functions, (QQ QF, FF)
    # then by the multipole order (0, 2, 4)
    #
    for j in  range(len(annotation)):
        i, order =annotation[j]
        eigenmode = eigenmodes[j]
        eigenmode[i].xi = numpy.array(eigenmode[i].xi, copy=True)

    return EigenModes(eigenmodes)

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))

