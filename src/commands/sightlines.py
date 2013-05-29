import numpy
import density
from args import pixeldtype0, pixeldtype1, writefill
from bresenham import clipline
from scipy.ndimage import map_coordinates
from scipy.stats import norm
import sharedmem
def main(A):
    """ raw guassian field"""
  
    print 'preparing large scale modes'
  
    delta0 = density.realize(A.power, A.Seed, 
                   A.NmeshCoarse, A.BoxSize, order=3)
  
    p = A.QSOpercentile
    print 'spawn and work on intermediate scales'
    
    cutoff = 0.5 * 2 * numpy.pi / A.BoxSize * A.NmeshCoarse

    with sharedmem.Pool() as pool:
      def work(i, j, k):
        # add in the small scale power
        delta1 = density.realize(A.power, A.SeedTable[i, j, k], 
                  A.NmeshQSO, A.BoxSize / A.Nrep, CutOff=cutoff)
          
        blendin(A, i, j, k, delta1, delta0)
        return delta1.sum(dtype='f8'), (delta1 **2).sum(dtype='f8')
  
      k0k1 = numpy.array(pool.starmap(work, list(A.yieldwork())))
    mean = k0k1[:, 0].sum() / (A.Nrep * A.NmeshQSO) ** 3
    std = (k0k1[:, 1].sum() / (A.Nrep * A.NmeshQSO) ** 3 - mean ** 2) ** 0.5
    print std, mean
    with sharedmem.Pool() as pool:
      def work(i, j, k):
        # add in the small scale power
        delta1 = density.realize(A.power, A.SeedTable[i, j, k],
                  A.NmeshQSO, A.BoxSize / A.Nrep, CutOff=cutoff)
          
        coarse = blendin(A, i, j, k, delta1, delta0)
        R = ((coarse - A.NmeshCoarse * 0.5) ** 2).sum(axis=0) ** 0.5 \
                * (A.BoxSize / A.NmeshCoarse)
        Rmin = A.cosmology.Dc(1 / (A.Redshift + 1)) * A.DH
        thresh = A.QSOpercentile(R)
        perc = norm.sf(delta1, loc=mean, scale=std).ravel()
        mask = perc < thresh
        mask &= R > Rmin
        QSOxyz = (coarse.T[mask] - 0.5 * A.NmeshCoarse) \
                * (A.BoxSize / A.NmeshCoarse)
        QSOR = R[mask]
        QSOdec = numpy.arcsin(QSOxyz[:, 2] / QSOR) * (180 / numpy.pi)
        QSOra = numpy.arctan2(QSOxyz[:, 1], QSOxyz[:, 0]) * (180 / numpy.pi)
        QSOz = 1. / A.cosmology.aback(QSOR / A.DH) - 1
        print "done ", i, j, k, len(QSOz)
        return numpy.array([QSOra, QSOdec, QSOz])
      rt = pool.starmap(work, list(A.yieldwork()))
      catelog = numpy.concatenate(rt, axis=-1)
      numpy.savetxt(A.datadir + '/QSOcatelog.txt', catelog.T)

def blendin(A, i, j, k, delta1, delta0):
    offset = numpy.array([i, j, k], dtype='f8')[:, None] * A.NmeshQSO
    # the grid on the QSO coordinate
    xyz = numpy.array(
            numpy.unravel_index(numpy.arange(delta1.size), delta1.shape))
    # convert to coarse coordinate
    xyz = xyz + (0.5 + offset)
    xyz = xyz / (1.0 * A.NmeshQSOEff) * A.NmeshCoarse
    coarse = map_coordinates(delta0, xyz, mode='wrap', order=3,
            prefilter=False)
    coarse.shape = delta1.shape
    delta1[...] += coarse
    return xyz

