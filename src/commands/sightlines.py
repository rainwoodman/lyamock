import numpy
import density
from args import pixeldtype
from bresenham import clipline
from scipy.ndimage import map_coordinates
from scipy.stats import norm
from density import lognormal
import sharedmem
def main(A):
    """ quasars identify quasars"""
  
    print 'preparing large scale modes'
  
    delta0, var0 = density.realize(A.power, A.Seed, 
                   A.NmeshCoarse, A.BoxSize, order=2, Kmax=A.KSplit)
  
    
    A.QSOdensity
    A.SurveyQSOdensity
    A.QSObias
    A.skymask
    A.cosmology.Dplus

    print 'bootstrap the variance'
    with sharedmem.Pool() as pool:
      def work(seed):
        # add in the small scale power
        delta1, var1 = density.realize(A.power, seed, 
                  A.NmeshQSO, A.BoxSize / A.Nrep, Kmin=A.KSplit)
        return var1
  
      # just do 16 small boxes to estimate the variance

      var1 = numpy.mean(pool.map(work, A.RNG.randint(0, 1<<21, size=16)))
    var = var1 + var0
    print 'total variance', var, 'coarse', var0, 'qso', var1
    std = var ** 0.5
#    std = 1.07401503829 
#    mean =  4.04737352692e-06
    print 'first run finished.'
    print 'std =', std

    print 'spawn and work on intermediate scales'
    output = file(A.datadir + '/QSOcatelog.raw', mode='w')
    with sharedmem.Pool() as pool:
      def work(i, j, k):
        # add in the small scale power
        delta1, var1 = density.realize(A.power, A.SeedTable[i, j, k],
                  A.NmeshQSO, A.BoxSize / A.Nrep, Kmin=A.KSplit)

        coarsepos = blendin(A, i, j, k, delta1, delta0)
          
        delta1 = delta1.reshape(-1)
        Rmin = A.cosmology.Dc(1 / (A.Zmin + 1)) * A.DH
        Rmax = A.cosmology.Dc(1 / (A.Zmax + 1)) * A.DH

        xyz = (coarsepos.T - 0.5 * A.NmeshCoarse) \
                * (A.BoxSize / A.NmeshCoarse)
        R = numpy.einsum('ij,ij->i', xyz, xyz) ** 0.5
        #DEC = numpy.arcsin(xyz[:, 2] / R)
        #RA = numpy.arctan2(xyz[:, 1], xyz[:, 0])

        rng = numpy.random.RandomState(A.SeedTable[i, j, k]) 
        u = rng.uniform(len(xyz))
        #apply the redshift and skymask selection
        mask = (R < Rmax) & (R > Rmin) & (A.skymask(xyz) > 0)
        if not mask.any():
            return [], [], [], [], []
        delta1 = delta1[mask]
        xyz = xyz[mask]
        R = R[mask]
        a = A.cosmology.aback(R / A.DH)
        Z = 1. / a - 1
        FakeZ = False # 0.24
        if FakeZ is not False:
            meandensity = A.QSOdensity(A.cosmology.Dc(1 / (FakeZ + 1.)) * A.DH)
            # override D to use z=2.0
            bias = 1.0
            D = A.cosmology.Dplus(1 / (FakeZ + 1.)) / A.cosmology.Dplus(1.0)
        else:
#        meandensity = A.QSOdensity(R)
            meandensity = A.SurveyQSOdensity(R)
            bias = A.QSObias(R) 
            D = A.cosmology.Dplus(a) / A.cosmology.Dplus(1.0)

        overdensity = bias * D * delta1
        # we do a lognormal to avoid negative number density
        lognormal(overdensity, std * (bias * D), out=overdensity)
        numberdensity = meandensity * (1 + overdensity)
        cellsize = A.BoxSize / A.NmeshQSOEff
        number = rng.poisson(numberdensity * cellsize ** 3)
        xyz = numpy.repeat(xyz, number, axis=0)
        xyz += rng.uniform(size=(len(xyz), 3), low=0, 
                high=cellsize)
        R = numpy.einsum('ij,ij->i', xyz, xyz) ** 0.5
        DEC = numpy.arcsin(xyz[:, 2] / R)
        RA = numpy.arctan2(xyz[:, 1], xyz[:, 0])
        a = A.cosmology.aback(R / A.DH)
        Z = 1. / a - 1
        return RA, DEC, Z, R, xyz
      def reduce(QSORA, QSODEC, QSOZ, R, xyz):
        if len(xyz) == 0: return 0
        QSORA *= 180 / numpy.pi
        QSODEC *= 180 / numpy.pi
        numpy.array([QSORA, QSODEC, QSOZ, R,
              xyz[:, 0], xyz[:, 1], xyz[:, 2]], dtype='f4').T.tofile(output)
        output.flush()
        return len(QSORA)
      NQSO = numpy.sum(pool.starmap(work, A.yieldwork(), reduce=reduce))
    DATA = numpy.memmap(A.datadir + '/QSOcatelog.raw', mode='r', dtype='f4',
            shape=(NQSO, 7))
    print len(DATA)
    numpy.savetxt(A.datadir + '/QSOcatelog.txt', DATA)
def blendin(A, i, j, k, delta1, delta0):
    offset = numpy.array([i, j, k], dtype='f8')[:, None] * A.NmeshQSO
    # the grid on the QSO coordinate
    xyz = numpy.array(
            numpy.unravel_index(numpy.arange(delta1.size), delta1.shape))
    # convert to coarse coordinate
    xyz = xyz + offset
    xyz = xyz / (1.0 * A.NmeshQSOEff) * A.NmeshCoarse
    fromcoarse = map_coordinates(delta0, xyz, mode='wrap', order=2,
            prefilter=False)

    fromcoarse.shape = delta1.shape
    delta1[...] += fromcoarse
    return xyz

