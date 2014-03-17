import numpy
from args import Config
from args import PowerSpectrum
from lib.bresenham import drawline
import sharedmem
from lib import density
from scipy.ndimage import spline_filter, map_coordinates

from smallscale import initlya
from sightlines import Sightlines

def main(A):
    global gaussian, shuffle, varlya, var1
    # gaussian are used for each subbox.
    # slaves are processes so they won't damage these variables
    # from the master.
    global powerspec
    global sightlines
    global samples
    global deltafield
    global objectidfield
    global velfield

    sightlines = Sightlines(A)
    powerspec = PowerSpectrum(A)
    gaussian = density.begin_irfftn((A.NmeshFine, A.NmeshFine, A.NmeshFine // 2 + 1),
            dtype=numpy.complex64)

    shuffle = density.build_shuffle((A.NmeshFine, A.NmeshFine, A.NmeshFine //2 + 1))
    
    # fine1 takes some time, so we do it async
    # while initing lya and estimating box layout.
    var0 = initcoarse(A)
    varlya = initlya(A)
    
    layout = A.layout(len(sightlines), chunksize=1024)

    Nsamples = sightlines.Nsamples.sum()
    print 'total number of pixels', Nsamples

    deltafield = numpy.memmap(A.DeltaField, mode='w+', 
                dtype='f4', shape=Nsamples)
    velfield = numpy.memmap(A.VelField, mode='w+', 
                dtype='f4', shape=Nsamples)
    objectidfield = numpy.memmap(A.ObjectIDField, mode='w+', 
                dtype='i4', shape=Nsamples)

    processors = [ 
            AddDelta,
            AddDisp(0),
            AddDisp(1),
            AddDisp(2),
        ]

    for proc in processors:
        proc.prepare(A)

    MemoryBytes = numpy.max([proc.MemoryBytes for proc in processors])

    np = int((sharedmem.total_memory() - 1024 ** 3) // MemoryBytes)
    np = numpy.min([sharedmem.cpu_count(), np])
    print 'spawn and work, with ', np, 'slaves', \
            'each use', MemoryBytes /1024.**2, 'MB'

    var1list = []
    with sharedmem.Pool(np=np) as pool:
        def work(i, j, k):
            box = layout[i, j, k]
            gaussian_save = numpy.empty_like(gaussian)
            # gaussian field will be reused and it 
            # takes time to generate them
            density.gaussian(gaussian_save, shuffle, 
                    A.SeedTable[i, j, k])
            var = None
            for cls in processors:
                proc = cls(box, gaussian_save)
                N = 0
                for chunk in box:
                    N += proc.visit(chunk)
                if hasattr(proc, 'var1'):
                    var = proc.var1
                # free memory
                del proc 
                if N == 0: 
                    # no pixels
                    # No need to work on other processors
                    break
            print 'done', i, j, k, N
            return var
        def reduce(v1):
            if v1 is not None:
                var1list.append(v1)
        pool.map(work, A.yieldwork(), reduce=reduce, star=True)
    for field in Visitor.M:
        Visitor.M[field].flush()

    var1 = numpy.mean(var1list)
    print 'gaussian-variance is', var0 + var1 + varlya
    numpy.savetxt(A.datadir + '/gaussian-variance.txt', [var0 + var1 + varlya])


def initcoarse(A):
    global delta0, disp0
    delta0, var0 = density.realize(powerspec, 
                   A.Seed, 
                   A.NmeshCoarse, 
                   A.BoxSize,
                   Kmax=A.KSplit)
    delta0 = spline_filter(delta0, order=4, output=numpy.dtype('f4'))

    disp0 = numpy.empty((3, A.NmeshCoarse, 
        A.NmeshCoarse, A.NmeshCoarse), dtype='f4')
    for d in range(3):
        disp0[d], junk = density.realize(powerspec, 
                 A.Seed, A.NmeshCoarse,
                 A.BoxSize, disp=d,
                 Kmax=A.KSplit)
        disp0[d] = spline_filter(disp0[d], order=4, output=numpy.dtype('f4')) 

    print 'coarse field var', var0
    return var0

class Visitor(object):
    M = {}
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
    def prepare(cls, A):
        Visitor.prepare(cls, A)
        cls.MemoryBytes = A.NmeshFine ** 3 * 8 * 2

    def __init__(self, box, gaussian_save):
        Visitor.__init__(self, box)
        A = self.A
        gaussian[...] = gaussian_save
        if True:
            delta1, junk = \
                density.realize(powerspec, 
                   None, 
                   A.NmeshFine, 
                   A.BoxSize / A.Nrep, 
                   Kmin=A.KSplit,
                   gaussian=gaussian)
            self.var1 = delta1.var()
            self.delta1 = spline_filter(delta1, order=4, 
                    output=numpy.dtype('f4'))
        else:
            self.delta1 = numpy.empty([A.NmeshFine, ] * 3)
            self.var1 = 1.0

        self.RNG = numpy.random.RandomState(
               A.SeedTable[box.i, box.j, box.k] 
            )

    def visit(self, chunk):
        xyz, objectid, t = self.getpixels(chunk)
        if len(xyz) == 0: 
            return 0
        chunk.empty = False
        xyzcoarse = self.getcoarse(xyz)
        d = map_coordinates(delta0, xyzcoarse.T, 
                mode='wrap', order=4, prefilter=False)

        xyzfine = self.getfine(xyz)
        d[:] += map_coordinates(self.delta1, xyzfine.T, 
                mode='wrap', order=4, prefilter=False)
        N = len(xyzfine)
        d[:] += self.RNG.normal(scale=varlya ** 0.5, size=N)
        deltafield[t] = d
        objectidfield[t] = numpy.int32(objectid)
        return len(xyz)

def AddDisp(d):
  class AddDisp(Visitor):
    @classmethod
    def prepare(cls, A):
        Visitor.prepare(cls, A)
        cls.MemoryBytes = A.NmeshFine ** 3 * 8 * 2
    def __init__(self, box, gaussian_save):
        Visitor.__init__(self, box)
        A = self.A
        dir = sightlines.dir
        gaussian[...] = gaussian_save
        self.disp1, junk = \
            density.realize(powerspec,
                None, 
                A.NmeshFine,
                A.BoxSize / A.Nrep, disp=d, Kmin=A.KSplit,
                gaussian=gaussian
                )
        self.disp1 = spline_filter(self.disp1, 
                order=4, output=numpy.dtype('f4'))
    def visit(self, chunk):
        xyz, objectid, t = self.getpixels(chunk)
        if len(xyz) == 0: return 0

        xyzcoarse = self.getcoarse(xyz)
        xyzfine = self.getfine(xyz)
        proj = sightlines.dir[objectid, d]
        disp = map_coordinates(disp0[d], 
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
