import numpy
import sharedmem
from common import Config
from common import PowerSpectrum
from common import Sightlines
from common import QSODensityModel
from common import QSOBiasModel
from common import Skymask

from scipy.ndimage import map_coordinates, spline_filter
from lib.lazy import Lazy
from lib import density

# this code makes the sightlines
from scipy.stats import norm

def main(A):
    """ quasars identify quasars"""
    print 'preparing large scale modes'

    global shuffle, gaussian, powerspec
    global var
    shuffle = density.build_shuffle((A.NmeshQSO, A.NmeshQSO, A.NmeshQSO //2 + 1))
    gaussian = density.begin_irfftn((A.NmeshQSO, A.NmeshQSO, A.NmeshQSO // 2 + 1),
            dtype=numpy.complex64)
    powerspec = PowerSpectrum(A)
    var0 = initcoarse(A)
    var1 = initqso1(A)
    var = var1 + var0
    bias = QSOBiasModel(A)(1/3.5) 
    D = A.cosmology.D(1/3.5)
    print 'total variance', var, 'coarse', var0, 'qso', var1
    print 'variance of QSO over density', var * (D * bias)** 2
#    std = var ** 0.5
#    std = 1.07401503829 
#    mean =  4.04737352692e-06
#    print 'first run finished.'
#    print 'std =', std
#   std is not used because we do not use the log normal 
    a = 1.0 / 3.0
    SurveyQSOdensity = QSODensityModel(A)
    qsonumberdensity = SurveyQSOdensity(a)
    b = 500.0
    percentile = b * qsonumberdensity * (A.BoxSize / A.Nrep / A.NmeshQSO)** 3

    sigma = norm.ppf(1.0 - percentile)
    print percentile, sigma
    numpy.savetxt(A.datadir + '/qso-variance.txt', [var])

    layout = A.layout(A.NmeshQSO ** 3, 1024 * 128)

    # purge the file
    output = file(A.QSOCatelog, mode='w')
    output.close()

    Visitor.prepare(A, var0)

#    sharedmem.set_debug(True)
    print 'spawn and work on intermediate scales'
    with sharedmem.Pool() as pool:
        def work(i, j, k):
            box = layout[i, j, k]
            proc = Visitor(box)
            N = 0
            for chunk in box:
                QSOs = proc.visit(chunk)
                with pool.critical:
                    with file(A.QSOCatelog, mode='a') as output:
                        raw = numpy.empty(len(QSOs), dtype=Sightlines.dtype)
                        raw['RA'] = QSOs.RA * 180 / numpy.pi
                        raw['DEC'] = QSOs.DEC * 180 / numpy.pi
                        raw['Z_RED'] = QSOs.Z
                        raw['Z_REAL'] = QSOs.Z
                        raw.tofile(output)
                        output.flush()
                N += len(QSOs)
            return N
                
        NQSO = numpy.sum(pool.map(work, A.yieldwork(), star=True))

    sightlines = Sightlines(A)
    print sightlines.LogLamMax, sightlines.LogLamMin
    print sightlines.LogLamGridIndMax, sightlines.LogLamGridIndMin

def initcoarse(A):
    global delta0
    delta0, var0 = density.realize(powerspec, 
                   A.Seed, 
                   A.NmeshCoarse, 
                   A.BoxSize,
                   Kmax=A.KSplit)
    delta0 = spline_filter(delta0, order=4, output=numpy.dtype('f4'))
    return var0

def initqso1(A):
    print 'bootstrap the variance'
    def kernel(kx, ky, kz, k):
        f2 = 1 / (1 + (A.QSOScale * k) ** 2)
        #f2 = numpy.exp(- (A.QSOScale * k) ** 2)
        return f2
    with sharedmem.Pool() as pool:
      def work(seed):
        density.gaussian(gaussian, shuffle, seed)
        # add in the small scale power
        delta1, var1 = density.realize(powerspec, None, 
                  A.NmeshQSO, A.BoxSize / A.Nrep, Kmin=A.KSplit,
                  kernel=kernel,
                  gaussian=gaussian)
        return var1
  
      # just do 16 small boxes to estimate the variance
      var1 = numpy.mean(pool.map(work, A.RNG.randint(0, 1<<21, size=1)))
    return var1


class Visitor(object):
    @classmethod
    def prepare(cls, A, var0):
        cls.A = A
        cls.var0 = var0
        cls.Dplus = staticmethod(A.cosmology.Dplus)
        cls.Dc = staticmethod(A.cosmology.Dc)

        cls.skymask = staticmethod(Skymask(A))
        cls.SurveyQSOdensity = staticmethod(QSODensityModel(A))
        cls.QSObias = staticmethod(QSOBiasModel(A))

    def __init__(self, box):
        A = self.A
        self.box = box
        density.gaussian(gaussian, shuffle, A.SeedTable[box.i, box.j, box.k])
        def kernel(kx, ky, kz, k):
            f2 = 1 / (1 + (A.QSOScale * k) ** 2)
            #f2 = numpy.exp(- (A.QSOScale * k) ** 2)
            return f2
        delta1, var1 = density.realize(powerspec, 
              None,
              A.NmeshQSO, A.BoxSize / A.Nrep, Kmin=A.KSplit,
              kernel=kernel,
              gaussian=gaussian)
        self.delta = delta1
        self.Rmin = self.Dc(1 / (A.Zmin + 1)) * A.DH
        self.Rmax = self.Dc(1 / (A.Zmax + 1)) * A.DH
        self.rng = numpy.random.RandomState(A.SeedTable[box.i, box.j, box.k]) 
        self.cellsize = A.BoxSize / (A.NmeshQSO * A.Nrep)
        self.var1 = var1

    def getcoarse(self, xyzqso):
        xyz = xyzqso + self.box.REPoffset * self.A.NmeshQSO 
        return 1.0 * xyz * self.A.NmeshCoarse / (self.A.NmeshQSO * self.A.Nrep)

    def getcenter(self, xyzcoarse):
        return xyzcoarse / self.A.NmeshCoarse * self.A.BoxSize - self.A.BoxSize * 0.5

    def selectpixels(self, xyz, delta):
        R = numpy.einsum('ij,ij->i', xyz, xyz) ** 0.5
        u = self.rng.uniform(len(xyz))
        #apply the redshift and skymask selection
        mask = (R < self.Rmax) & (R > self.Rmin) & (self.skymask(xyz) > 0)
        delta = delta[mask]
        xyz = xyz[mask]
        R = R[mask]
        a = self.Dc.inv(R / self.A.DH)
        return xyz, a, delta

    def getNqso(self, a, delta):
#        bias = self.QSObias(a) 
        #always use z=2.0
        A = self.A
        D = A.cosmology.D(a)
        bias = self.QSObias(a)
        var = self.var1 + self.var0
        deltaLN = numpy.exp(bias * D * delta - (bias ** 2 * D ** 2 * var) * 0.5)
        qsonumberdensity = deltaLN * self.SurveyQSOdensity(a) 

        Nqso = qsonumberdensity * self.cellsize ** 3
        Nqso = self.rng.poisson(Nqso)
        #percentile = b * qsonumberdensity * self.cellsize ** 3

        #var = self.var1 + self.var0
        #std = var ** 0.5
        #mask = delta / std > norm.ppf(1.0 - percentile)
        # save the number of qsos, for non-host it is zero
        #Nexpected = numpy.zeros_like(delta)
        #Nexpected[mask] = 1.0 / b
        #Nqso = numpy.zeros_like(delta, 'int32')
        #Nqso[...] = self.rng.poisson(Nexpected)

        return Nqso

    def makeqso(self, xyz, Nqso):
        xyz = numpy.repeat(xyz, Nqso, axis=0)
        xyz += self.rng.uniform(size=(len(xyz), 3), low=0, 
                    high=self.cellsize)
        R = numpy.einsum('ij,ij->i', xyz, xyz) ** 0.5
        DEC = numpy.arcsin(xyz[:, 2] / R)
        RA = numpy.arctan2(xyz[:, 1], xyz[:, 0])
        a = self.Dc.inv(R / self.A.DH)
        Z = 1. / a - 1
        return numpy.rec.fromarrays([R, DEC, RA, Z],
                names=['R', 'DEC', 'RA', 'Z'])

    def visit(self, chunk):
        sl = chunk.getslice()
        start, end, step = chunk.getslice().indices(self.A.NmeshQSO ** 3)
        linear = numpy.arange(start, end, step)
        xyzqso = numpy.array(numpy.unravel_index(linear, (self.A.NmeshQSO,) * 3)).T
        xyzcoarse = self.getcoarse(xyzqso)
        delta = self.delta.take(linear) 
        delta += map_coordinates(delta0,
                xyzcoarse.T, mode='wrap', order=4,
                prefilter=False)

        xyz = self.getcenter(xyzcoarse)    
        xyz, a, delta = self.selectpixels(xyz, delta)
        Nqso = self.getNqso(a, delta)
        return self.makeqso(xyz, Nqso)

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
