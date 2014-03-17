import numpy
import sharedmem
from args import Config
from args import PowerSpectrum

from scipy.ndimage import map_coordinates, spline_filter
from lib.lazy import Lazy
from lib import density

# this code makes the sightlines
# also provides Sightlines(config) that returns
# the sightline catalogue

sightlinedtype=numpy.dtype([('RA', 'f8'), 
                             ('DEC', 'f8'), 
                             ('Z_RED', 'f8'),
                             ('Z_REAL', 'f8'),
                             ])

class Sightlines(object):
    def __init__(self, config):
        self.data = numpy.fromfile(config.QSOCatelog, 
                dtype=sightlinedtype)
        self.Z_REAL = self.data['Z_REAL']
        self.DEC = self.data['DEC']
        self.RA = self.data['RA']
        self.config = config

        # Z_RED is writable!
        data2 = numpy.memmap(config.QSOCatelog, dtype=sightlinedtype, mode='r+')
        self.Z_RED = data2['Z_RED']

    def __len__(self):
        return len(self.data)

    @Lazy
    def SampleOffset(self):
        rt = numpy.empty(len(self), dtype='intp')
        rt[0] = 0
        rt[1:] = numpy.cumsum(self.Nsamples)[:-1]
        return rt

    @Lazy
    def R1(self):
        cosmology = self.config.cosmology
        R1 = cosmology.Dc(1 / (self.Zmin + 1))
        return R1

    @Lazy
    def Nsamples(self):
        cosmology = self.config.cosmology
        R1 = cosmology.Dc(1 / (self.Zmin + 1))
        R2 = cosmology.Dc(1 / (self.Zmax + 1))
        return numpy.int32((R2 - R1) * self.config.DH / self.config.LogNormalScale)

    @Lazy
    def x1(self):
        cosmology = self.config.cosmology
        return self.dir * cosmology.Dc(1 / (self.Zmin + 1))[:, None] * self.config.DH

    @Lazy
    def x2(self):
        cosmology = self.config.cosmology
        return self.dir * cosmology.Dc(1 / (self.Zmax + 1))[:, None] * self.config.DH

    @Lazy
    def dir(self):
        dir = numpy.empty((len(self), 3))
        dir[:, 0] = numpy.cos(self.RA) * numpy.cos(self.DEC)
        dir[:, 1] = numpy.sin(self.RA) * numpy.cos(self.DEC)
        dir[:, 2] = numpy.sin(self.DEC)
        return dir
    @Lazy
    def Zmax(self):
        return (self.Z_REAL + 1) * 1216. / 1216 - 1
    @Lazy
    def Zmin(self):
        return (self.Z_REAL + 1) * 1026. / 1216 - 1

    @Lazy
    def LogLamMin(self):
        return numpy.log10((self.Zmin + 1) * 1216.)

    @Lazy
    def LogLamMax(self):
        return numpy.log10((self.Zmax + 1) * 1216.)

    @Lazy
    def LogLamGridIndMin(self):
        rt = numpy.searchsorted(self.config.LogLamGrid, 
                    self.LogLamMin, side='left')
        return rt
    @Lazy
    def LogLamGridIndMax(self):
        rt = numpy.searchsorted(self.config.LogLamGrid, 
                    self.LogLamMax, side='right')
        # probably no need to care if it goes out of limit. 
        # really should have used 1e-4 binning than the search but
        # need to deal with the clipping later on.
        return rt
    @Lazy
    def Npixels(self):
        return self.LogLamGridIndMax - self.LogLamGridIndMin
    @Lazy
    def PixelOffset(self):
        rt = numpy.empty(len(self), dtype='intp')
        rt[0] = 0
        rt[1:] = numpy.cumsum(self.Npixels)[:-1]
        return rt

def main(A):
    """ quasars identify quasars"""
    print 'preparing large scale modes'

    global shuffle, gaussian, powerspec
    shuffle = density.build_shuffle((A.NmeshQSO, A.NmeshQSO, A.NmeshQSO //2 + 1))
    gaussian = density.begin_irfftn((A.NmeshQSO, A.NmeshQSO, A.NmeshQSO // 2 + 1),
            dtype=numpy.complex64)
    powerspec = PowerSpectrum(A)
    var0 = initcoarse(A)
#    var1 = initqso1(A)
#    var = var1 + var0
#    print 'total variance', var, 'coarse', var0, 'qso', var1
#    std = var ** 0.5
#    std = 1.07401503829 
#    mean =  4.04737352692e-06
#    print 'first run finished.'
#    print 'std =', std
#   std is not used because we do not use the log normal 
    std = 1.0

    layout = A.layout(A.NmeshQSO ** 3, 1024 * 128)

    # purge the file
    output = file(A.QSOCatelog, mode='w')
    output.close()

    Visitor.prepare(A, std)

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
                        raw = numpy.empty(len(QSOs), dtype=sightlinedtype)
                        raw['RA'] = QSOs.RA * 180 / numpy.pi
                        raw['DEC'] = QSOs.DEC * 180 / numpy.pi
                        raw['Z_VI'] = -1.0
                        raw['Z_REAL'] = QSOs.Z
                        raw.tofile(output)
                        output.flush()
                N += len(QSOs)
            return N
                
        NQSO = numpy.sum(pool.map(work, A.yieldwork(), star=True))

    sightlines = Sightlines(A)
    print sightlines.Zmax, sightlines.Zmin
    print sightlines.LogLamMax, sightlines.LogLamMin
    print sightlines.LogLamGridIndMax, sightlines.LogLamGridIndMin

"""
    import chealpy
    catelog = A.P('QSOcatelog', memmap='r+', dtype=sightlinedtype)

    key = chealpy.ang2pix_nest(128, 
            0.5 * numpy.pi - catelog['DEC'] * (numpy.pi / 180),
            catelog['RA'] * (numpy.pi / 180))

    print 'sorting', len(catelog), 'quasars and assigning fibers'
    arg = catelog['Z_REAL'].argsort()
    catelog[:] = catelog[arg]
    assignfiber(A, catelog)
    print 'writing', len(catelog), 'quasars'
"""
    
def assignfiber(A, catelog):
    ind = A.fibers['Z_VI'].searchsorted(catelog['Z_VI'])
    begin = numpy.concatenate((
        [0],
        (ind[1:] > ind[:-1]).nonzero()[0] + 1,
        [len(ind)]))
    l = numpy.diff(begin)
    seq = numpy.arange(len(ind)) - begin[:-1].repeat(l)
    ind = (ind + seq).clip(0, len(A.fibers) - 1)
    
    catelog['refMJD'] = A.fibers['MJD'][ind]
    catelog['refPLATE'] = A.fibers['PLATE'][ind]
    catelog['refFIBERID'] = A.fibers['FIBERID'][ind]

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
    with sharedmem.Pool() as pool:
      def work(seed):
        density.gaussian(gaussian, shuffle, seed)
        # add in the small scale power
        delta1, var1 = density.realize(powerspec, None, 
                  A.NmeshQSO, A.BoxSize / A.Nrep, Kmin=A.KSplit,
                  gaussian=gaussian)
        return var1
  
      # just do 16 small boxes to estimate the variance
      var1 = numpy.mean(pool.map(work, A.RNG.randint(0, 1<<21, size=16)))
    return var1


class Visitor(object):
    @classmethod
    def prepare(cls, A, std):
        cls.A = A
        cls.std = std
        cls.skymask = staticmethod(A.skymask)
        cls.Dplus = staticmethod(A.cosmology.Dplus)
        cls.Dc = staticmethod(A.cosmology.Dc)
        cls.SurveyQSOdensity = staticmethod(A.SurveyQSOdensity)
        cls.QSObias = staticmethod(A.QSObias)

    def __init__(self, box):
        A = self.A
        self.box = box
        density.gaussian(gaussian, shuffle, A.SeedTable[box.i, box.j, box.k])
        delta1, var1 = density.realize(powerspec, 
              None,
              A.NmeshQSO, A.BoxSize / A.Nrep, Kmin=A.KSplit,
              gaussian=gaussian)
        self.delta = delta1
        self.Rmin = self.Dc(1 / (A.Zmin + 1)) * A.DH
        self.Rmax = self.Dc(1 / (A.Zmax + 1)) * A.DH
        self.rng = numpy.random.RandomState(A.SeedTable[box.i, box.j, box.k]) 
        self.cellsize = A.BoxSize / A.NmeshQSOEff

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
        return xyz, R, a, delta

    def getNqso(self, R, a, delta):
        meandensity = self.SurveyQSOdensity(R)
        bias = self.QSObias(R) 
        D = self.Dplus(a) / self.Dplus(1.0)

        overdensity = bias * D * delta
#        # we do a lognormal to avoid negative number density
#       lognormal messes it up
#        lognormal(overdensity, self.std * (bias * D), out=overdensity)
        numberdensity = meandensity * (1 + overdensity)
        numberdensity[numberdensity < 0] = 0
        Nqso = self.rng.poisson(numberdensity * self.cellsize ** 3)
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
        xyz, R, a, delta = self.selectpixels(xyz, delta)
        Nqso = self.getNqso(R, a, delta)
        return self.makeqso(xyz, Nqso)

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
