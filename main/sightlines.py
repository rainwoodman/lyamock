import numpy
import sharedmem
from common import Config
from common import PowerSpectrum
from common import Sightlines
from common import QSODensityModel
from common import QSOBiasModel
from common import Skymask


from lib.lazy import Lazy
from lib import density2
from lib.ndimage import map_coordinates, spline_filter

# this code makes the sightlines
from scipy.stats import norm

def main(A):
    """ quasars identify quasars"""
    print 'preparing large scale modes'

    powerspec = PowerSpectrum(A)
    den1 = density2.Density(A.NmeshQSO, power=powerspec,
            BoxSize=A.BoxSize/A.Nrep,
            Kmax=A.Kmax,
            Kmin=A.KSplit)
            
    print 'init coarse'
    delta0, var0 = initcoarse(A, powerspec)
    print 'done coarse'

    layout = A.layout(A.NmeshQSO ** 3, 1024 * 1024)

    Visitor.prepare(A, delta0, var0)

#    sharedmem.set_debug(True)
    print 'spawn and work on intermediate scales'

    # purge the file
    output = file(A.QSOCatelog, mode='w')
    output.close()

    with sharedmem.Pool() as pool:
        def work(i, j, k):
            box = layout[i, j, k]
            proc = Visitor(box, den1)
            N = 0
            for chunk in box:
                QSOs = proc.visit(chunk)
                with pool.critical:
                    with file(A.QSOCatelog, mode='a') as output:
                        raw = numpy.empty(len(QSOs), dtype=Sightlines.dtype)
                        raw['RA'] = QSOs.RA
                        raw['DEC'] = QSOs.DEC
                        raw['Z_RED'] = QSOs.Z
                        raw['Z_REAL'] = QSOs.Z
                        raw.tofile(output)
                        output.flush()
            if proc.mom[0] > 0:
                print proc.mom[0], proc.mom[1] / proc.mom[0], \
                    (proc.mom[2] / proc.mom[0]), proc.var

                N += len(QSOs)
            return N
                
        NQSO = numpy.sum(pool.map(work, A.yieldwork(), star=True))

    sightlines = Sightlines(A)
    print sightlines.LogLamMax, sightlines.LogLamMin
    print sightlines.LogLamGridIndMax, sightlines.LogLamGridIndMin
    print len(sightlines), QSODensityModel(A).Nqso

def initcoarse(A, powerspec):
    den0 = density2.Density(A.NmeshCoarse,
            power=powerspec,
            BoxSize=A.BoxSize,
            Kmax=A.KSplit)

    den0.fill(seed=A.Seed, kernel=None)
    delta0 = den0.realize()
    var0 = delta0.var(dtype='f8')
    delta0 = spline_filter(delta0, order=4, output=numpy.dtype('f4'))
    return delta0, var0

class Visitor(object):
    @classmethod
    def prepare(cls, A, delta0, var0):
        cls.A = A
        cls.var0 = var0
        cls.delta0 = delta0
        cls.Dplus = staticmethod(A.cosmology.Dplus)
        cls.Dc = staticmethod(A.cosmology.Dc)

        cls.skymask = staticmethod(Skymask(A))
        cls.SurveyQSOdensity = staticmethod(QSODensityModel(A))
        cls.QSObias = staticmethod(QSOBiasModel(A))

    def __init__(self, box, den1):
        A = self.A
        self.box = box
        self.den1 = den1
        i, j, k = box.i, box.j, box.k
        self.init = False
        self.rng = numpy.random.RandomState(A.SeedTable[i, j, k]) 
        self.cellsize = A.BoxSize / (A.NmeshQSO * A.Nrep)
        self.Rmin = self.Dc(1 / (A.Zmin + 1)) * A.DH
        self.Rmax = self.Dc(1 / (A.Zmax + 1)) * A.DH
        self.mom = [0., 0., 0.]

    @Lazy
    def var(self):
        return self.var0 + self.delta1.var(dtype='f8')

    @Lazy
    def delta1(self):
        A = self.A
        box = self.box
        i, j, k = box.i, box.j, box.k
        self.den1.fill(seed=A.SeedTable[i, j, k], kernel=None)

        delta1 = self.den1.realize()
        return delta1

    def getcoarse(self, xyzqso):
        xyz = xyzqso + self.box.REPoffset * self.A.NmeshQSO 
        return xyz * (1.0 * self.A.NmeshCoarse / (self.A.NmeshQSO * self.A.Nrep))

    def getcenter(self, xyzcoarse):
        return xyzcoarse * (1.0 / self.A.NmeshCoarse * self.A.BoxSize) - self.A.BoxSize * 0.5

    def getNqso(self, a, delta):
#        bias = self.QSObias(a) 
        #always use z=2.0
        A = self.A
        D = A.cosmology.D(a)
        bias = self.QSObias(a)
        deltaLN = numpy.exp(bias * D * delta - (bias ** 2 * D ** 2 * self.var) * 0.5)
        deltaLN[deltaLN > 40.0] = 40.0 # remove 40 sigma peaks
        qsonumberdensity = deltaLN * self.SurveyQSOdensity(a) 
        self.mom[0] += len(deltaLN)
        self.mom[1] += delta.sum(dtype='f8')
        self.mom[2] += (delta ** 2).sum(dtype='f8')

        Nqso = qsonumberdensity * self.cellsize ** 3
        Nqso = self.rng.poisson(Nqso)

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
        xyz = self.getcenter(xyzcoarse)    
        R2 = numpy.einsum('ij,ij->i', xyz, xyz)
        u = self.rng.uniform(len(xyz))
        #apply the redshift and skymask selection
        mask = (R2 < self.Rmax ** 2) & (R2 > self.Rmin ** 2) & (self.skymask(xyz) > 0)

        if self.box.i == 0 and self.box.j == 0 and self.box.k == 0:
            numpy.save('delta1.npy', self.delta1)

        if not mask.any():
            return numpy.rec.fromarrays([[], [], [], []],
                    names=['R', 'DEC', 'RA', 'Z'])
            # avoid accessing self.delta1
            # which does the fft stuff.

        linear = linear[mask]
        xyz = xyz[mask]
        xyzcoarse = xyzcoarse[mask]
        xyzqso = xyzqso[mask]
        R = R2[mask] ** 0.5

        delta = self.delta1.take(linear) 

        delta += map_coordinates(self.delta0,
                xyzcoarse.T, mode='wrap', order=4,
                prefilter=False)
        a = self.Dc.inv(R / self.A.DH)
        Nqso = self.getNqso(a, delta)
        return self.makeqso(xyz, Nqso)

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
