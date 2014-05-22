import numpy
from common import Config
from common import PowerSpectrum
from lib.bresenham import drawline
import sharedmem
from lib import density2
from lib.lazy import Lazy
from lib.ndimage import spline_filter, map_coordinates

from smallscale import initlya
from sightlines import Sightlines

dispkernel = [
        lambda kx, ky, kz, k: 1j * kx * k ** -2,
        lambda kx, ky, kz, k: 1j * ky * k ** -2,
        lambda kx, ky, kz, k: 1j * kz * k ** -2
]

def main(A):
    # gaussian are used for each subbox.
    # slaves are processes so they won't damage these variables
    # from the master.
    global sightlines

    global deltafield
    global objectidfield
    global velfield

    sightlines = Sightlines(A)
    powerspec = PowerSpectrum(A)
    
    # fine1 takes some time, so we do it async
    # while initing lya and estimating box layout.
    delta0, var0, disp0 = initcoarse(A, powerspec)
    varlya = initlya(A)

    den1 = density2.Density(A.NmeshFine, 
            Kmax=A.Kmax,
            Kmin=A.KSplit, power=powerspec, 
            BoxSize=A.BoxSize / A.Nrep)
    
    layout = A.layout(len(sightlines), chunksize=1024)

    Nsamples = sightlines.Nsamples.sum()
    print 'total number of pixels', Nsamples

    deltafield = sharedmem.empty(shape=Nsamples, dtype='f4')
    velfield = sharedmem.empty(shape=Nsamples, dtype='f4')
    objectidfield = sharedmem.empty(shape=Nsamples, dtype='i4')
    
    processors = [ 
            (AddDelta, delta0),
            (AddDisp(0), disp0[0]),
            (AddDisp(1), disp0[1]), 
            (AddDisp(2), disp0[2]),
        ]

    for proc, d0 in processors:
        proc.prepare(A, d0)

    MemoryBytes = numpy.max([proc.MemoryBytes for proc, d0 in processors])

    np = int((sharedmem.total_memory() - 1024 ** 3) // MemoryBytes)
    np = numpy.min([sharedmem.cpu_count(), np])
    print 'spawn and work, with ', np, 'slaves', \
            'each use', MemoryBytes /1024.**2, 'MB'

    var1list = []
    with sharedmem.Pool(np=np) as pool:
        def work(i, j, k):
            box = layout[i, j, k]
            var = None
            for cls, d0 in processors:
                proc = cls(box, den1, varlya)
                N = 0
                for chunk in box:
                    N += proc.visit(chunk)
                if cls is AddDelta:
                    var1 = proc.var1

                # free memory
                del proc 

                if N == 0: 
                    # no pixels
                    # No need to work on other processors
                    break
            print 'done', i, j, k, N, var1
            return var1
        def reduce(v1):
            if v1 is not None:
                var1list.append(v1)
        pool.map(work, A.yieldwork(), reduce=reduce, star=True)


    deltafield.tofile(A.DeltaField)
    velfield.tofile(A.VelField)
    objectidfield.tofile(A.ObjectIDField)

    D2 = A.cosmology.Dplus(1 / 3.0) / A.cosmology.Dplus(1.0)
    D3 = A.cosmology.Dplus(1 / 4.0) / A.cosmology.Dplus(1.0)
    var1 = numpy.nanmean(var1list)
    var = var0 + var1 + varlya
    print 'gaussian-variance is', var
    numpy.savetxt(A.datadir + '/gaussian-variance.txt', [var])
    print 'lya field', 'var', var
    print 'growth factor at z=2.0, 3.0', D2, D3
    print 'lya variance adjusted to z=2.0, z=3.0', D2 ** 2 * var, D3 **2 * var


def initcoarse(A, powerspec):
    den0 = density2.Density(A.NmeshCoarse, 
            power=powerspec,
            BoxSize=A.BoxSize,
            Kmax=A.KSplit)

    den0.fill(seed=A.Seed, kernel=None)

    delta0 = den0.realize()
    var0 = delta0.var(dtype='f8')
    delta0 = spline_filter(delta0, order=4, output=numpy.dtype('f4'))
    
    disp0 = numpy.empty((3, A.NmeshCoarse, 
        A.NmeshCoarse, A.NmeshCoarse), dtype='f4')
    for d in range(3):
        den0.fill(seed=A.Seed, kernel=dispkernel[d])
        disp0[d] = den0.realize()
        disp0[d] = spline_filter(disp0[d], order=4, output=numpy.dtype('f4')) 

    print 'coarse field var', var0
    return delta0, var0, disp0

class Visitor(object):
    @staticmethod
    def prepare(cls, A):
        """ prepare is called before the fork """
        cls.A = A

    def __init__(self, box):
        """ init is called after the fork """
        self.box = box

    def visit(self, chunk):
        pass

    def getcoarse(self, xyz):
        return (xyz + self.A.BoxSize * 0.5) * (1.0 * self.A.NmeshCoarse / self.A.BoxSize)
    def getfine(self, xyz):
        return (xyz - self.box.origin[None, :]) * \
                (1.0 * self.A.NmeshFine / (self.A.BoxSize / self.A.Nrep))

    def getpixels(self, chunk):
        """ return_full is True, return xyzlya, objectid 
            False, return total number of pixels in chunk """
        s = chunk.getslice()
        result = \
            drawline(
                sightlines.x1[s], sightlines.x2[s], 
                self.A.LogNormalScale,
                self.box.origin, 
                self.box.origin + self.A.BoxSize / self.A.Nrep,
                return_full=True)
        xyz, objectid, t = result
        if len(xyz) != 0:
            Nsamples = sightlines.Nsamples[s]
            mask = (t < Nsamples[objectid]) & (t >= 0)
            xyz = xyz[mask]
            objectid = objectid[mask] + chunk.offset
            t = t[mask] + sightlines.SampleOffset[objectid]
        return xyz, objectid, t

class AddDelta(Visitor):
    @classmethod
    def prepare(cls, A, delta0):
        Visitor.prepare(cls, A)
        cls.delta0 = delta0
        cls.MemoryBytes = A.NmeshFine ** 3 * 8 * 2

    def __init__(self, box, den1, varlya):
        Visitor.__init__(self, box)
        A = self.A
        self.varlya = varlya
        self.var1 = numpy.nan
        self.RNG = numpy.random.RandomState(
               A.SeedTable[box.i, box.j, box.k] 
            )
        self.den1 = den1

    @Lazy
    def delta1(self):
        box = self.box
        i, j, k = box.i, box.j, box.k
        if True:
            self.den1.fill(seed=self.A.SeedTable[i, j, k], kernel=None)
            delta1 = self.den1.realize()
            self.var1 = delta1.var(dtype='f8')
            return spline_filter(delta1, order=4,
                    output=numpy.dtype('f4'))
        else:
            return numpy.zeros([A.NmeshFine, ] * 3, dtype='f4')

    def visit(self, chunk):
        xyz, objectid, t = self.getpixels(chunk)
        if len(xyz) == 0: 
            return 0
        chunk.empty = False
        xyzcoarse = self.getcoarse(xyz)
        d = map_coordinates(self.delta0, xyzcoarse.T, 
                mode='wrap', order=4, prefilter=False)

        xyzfine = self.getfine(xyz)
        d[:] += map_coordinates(self.delta1, xyzfine.T, 
                mode='wrap', order=4, prefilter=False)
        N = len(xyzfine)
        d[:] += self.RNG.normal(scale=self.varlya ** 0.5, size=N)
        deltafield[t] = d
        objectidfield[t] = numpy.int32(objectid)
        return len(xyz)

def AddDisp(d):
  class AddDisp(Visitor):
    @classmethod
    def prepare(cls, A, disp0):
        Visitor.prepare(cls, A)
        cls.MemoryBytes = A.NmeshFine ** 3 * 8 * 2
        cls.disp0 = disp0

    def __init__(self, box, den1, varlya):
        Visitor.__init__(self, box)
        self.den1 = den1
        A = self.A

    @Lazy
    def disp1(self):
        A = self.A
        box = self.box
        i, j, k = box.i, box.j, box.k
        if True:
            self.den1.fill(seed=A.SeedTable[i, j, k], kernel=dispkernel[d])
            delta1 = self.den1.realize()
            return spline_filter(delta1, order=4,
                    output=numpy.dtype('f4'))
        else:
            return numpy.zeros([A.NmeshFine, ] * 3, dtype='f4')

    def visit(self, chunk):
        xyz, objectid, t = self.getpixels(chunk)
        if len(xyz) == 0: return 0

        xyzcoarse = self.getcoarse(xyz)
        xyzfine = self.getfine(xyz)
        proj = sightlines.dir[objectid, d]
        disp = map_coordinates(self.disp0, 
                xyzcoarse.T,
                mode='wrap', order=4, prefilter=False)
        disp += map_coordinates(self.disp1, 
                xyzfine.T,
                mode='wrap', order=4, prefilter=False)
        velfield[t] += disp * proj
        return len(xyz)
  return AddDisp

if __name__ == '__main__':
    from sys import argv
    config = Config(argv[1])
    main(config)
