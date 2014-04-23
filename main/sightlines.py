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

    shuffle = density.build_shuffle((A.NmeshQSO, A.NmeshQSO, A.NmeshQSO //2 + 1))
    gaussian = density.begin_irfftn((A.NmeshQSO, A.NmeshQSO, A.NmeshQSO),
            dtype=numpy.complex64)
    powerspec = PowerSpectrum(A)
    var0 = initcoarse(A)

    layout = A.layout(A.NmeshQSO ** 3, 1024 * 1024)

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
            if proc.mom[0] > 0:
                print proc.mom[0], proc.mom[1] / proc.mom[0], \
                    (proc.mom[2] / proc.mom[0]), proc.var, proc.var0, proc.var1\

                N += len(QSOs)
            return N
                
        NQSO = numpy.sum(pool.map(work, A.yieldwork(), star=True))

    sightlines = Sightlines(A)
    print sightlines.LogLamMax, sightlines.LogLamMin
    print sightlines.LogLamGridIndMax, sightlines.LogLamGridIndMin
    print len(sightlines), SurveyQSOdensity.Nqso

def initcoarse(A):
    global delta0
    delta0, var0 = density.realize(powerspec, 
                   A.Seed, 
                   A.NmeshCoarse, 
                   A.BoxSize,
                   Kmax=A.KSplit)
    delta0 = spline_filter(delta0, order=4, output=numpy.dtype('f4'))
    return var0

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
        self.init = False
        self.rng = numpy.random.RandomState(A.SeedTable[box.i, box.j, box.k]) 
        self.cellsize = A.BoxSize / (A.NmeshQSO * A.Nrep)
        self.Rmin = self.Dc(1 / (A.Zmin + 1)) * A.DH
        self.Rmax = self.Dc(1 / (A.Zmax + 1)) * A.DH
        self.mom = [0., 0., 0.]

    def deferinit(self):
        A = self.A
        box = self.box
        density.gaussian(gaussian, shuffle, A.SeedTable[box.i, box.j, box.k])
        def kernel(kx, ky, kz, k):
            f2 = 1 / (1 + (A.QSOScale * k) ** 2)
            #f2 = numpy.exp(- (A.QSOScale * k) ** 2)
            return f2
        delta1, var1 = density.realize(powerspec, 
              None,
              A.NmeshQSO, A.BoxSize / A.Nrep, Kmin=A.KSplit,
        #      kernel=kernel,
              gaussian=gaussian)
        self.delta = delta1
        self.var1 = var1
        self.var = var1 + self.var0
    
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
        var = self.var1 + self.var0
        deltaLN = numpy.exp(bias * D * delta - (bias ** 2 * D ** 2 * var) * 0.5)
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
            self.deferinit()
            numpy.save('delta1.npy', self.delta)
            self.init = True

        if mask.any():
            if not self.init:
                self.deferinit()
                self.init = True
        else:
            return numpy.rec.fromarrays([[], [], [], []],
                    names=['R', 'DEC', 'RA', 'Z'])

        linear = linear[mask]
        xyz = xyz[mask]
        xyzcoarse = xyzcoarse[mask]
        xyzqso = xyzqso[mask]
        R = R2[mask] ** 0.5

        delta = self.delta.take(linear) 

        delta += map_coordinates(delta0,
                xyzcoarse.T, mode='wrap', order=4,
                prefilter=False)
        a = self.Dc.inv(R / self.A.DH)
        Nqso = self.getNqso(a, delta)
        return self.makeqso(xyz, Nqso)

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
