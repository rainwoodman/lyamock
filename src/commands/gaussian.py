import numpy
from bresenham import drawline
import sharedmem
import density
from scipy.ndimage import spline_filter, map_coordinates

def main(A):
    global gaussian, shuffle, varlya, var1
    # gaussian are used for each subbox.
    # slaves are processes so they won't damage these variables
    # from the master.
    A.power

    gaussian = density.begin_irfftn((A.NmeshFine, A.NmeshFine, A.NmeshFine // 2 + 1),
            dtype=numpy.complex64)

    shuffle = density.build_shuffle((A.NmeshFine, A.NmeshFine, A.NmeshFine //2 + 1))
    
    # fine1 takes some time, so we do it async
    # while initing lya and estimating box layout.
    var0 = initcoarse(A)
    varlya = initlya(A)
    
    layout = A.layout(len(A.sightlines), chunksize=1024)

    layout.Npixels = sharedmem.empty((A.Nrep,) * 3,
            dtype=('intp', (layout.Nchunks, )))

    Estimator.prepare(A)
    with sharedmem.MapReduce() as pool:
        def work(i, j, k):
            box = layout[i, j, k]
            proc = Estimator(box)
            for chunk in box:
                proc.visit(chunk)
        pool.map(work, A.yieldwork(), star=True)

    layout.PixelEnd = layout.Npixels.cumsum().reshape(layout.Npixels.shape)
    layout.PixelStart = layout.PixelEnd.copy()
    layout.PixelStart[0] = 0
    layout.PixelStart.flat[1:] = layout.PixelEnd.flat[:-1]

    processors = [ 
            GeneratePixels,
            AddDelta,
            AddDisp(2),
        ]
    if A.Geometry != 'Test':
        processors += [
            AddDisp(0),
            AddDisp(1),
            ]

    print layout.Npixels.sum()
    M = {}
    for proc in processors:
        proc.prepare(A, layout.Npixels.sum())

    MemoryBytes = numpy.max([proc.MemoryBytes for proc in processors])

    np = int((sharedmem.total_memory() - 1024 ** 3) // MemoryBytes)
    np = numpy.min([sharedmem.cpu_count(), np])
    print 'spawn and work, with ', np, 'slaves', \
            'each use', MemoryBytes /1024.**2, 'MB'

    var1list = []
    with sharedmem.Pool(np=np) as pool:
        def work(i, j, k):
            box = layout[i, j, k]
            if box.Npixels.sum() == 0: return
            gaussian_save = numpy.empty_like(gaussian)
            # gaussian field will be reused and it 
            # takes time to generate them
            density.gaussian(gaussian_save, shuffle, 
                    A.SeedTable[i, j, k])
            var = None
            for cls in processors:
                proc = cls(box, gaussian_save)
                for chunk in box:
                    proc.visit(chunk)
                if hasattr(proc, 'var1'):
                    var = proc.var1
                # free memory
                del proc 
            print 'done', i, j, k
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
    delta0, var0 = density.realize(A.power, 
                   A.Seed, 
                   A.NmeshCoarse, 
                   A.BoxSize,
                   Kmax=A.KSplit)
    delta0 = spline_filter(delta0, order=4, output=numpy.dtype('f4'))

    disp0 = numpy.empty((3, A.NmeshCoarse, 
        A.NmeshCoarse, A.NmeshCoarse), dtype='f4')
    for d in range(3):
        disp0[d], junk = density.realize(A.power, 
                 A.Seed, A.NmeshCoarse,
                 A.BoxSize, disp=d,
                 Kmax=A.KSplit)
        disp0[d] = spline_filter(disp0[d], order=4, output=numpy.dtype('f4')) 

    print 'coarse field var', var0
    return var0

def initlya(A):
    print 'preparing LyaBoxes'
  
    # fill the lya resolution small boxes
    deltalya = sharedmem.empty((A.NLyaBox, 
        A.NmeshLyaBox, A.NmeshLyaBox, A.NmeshLyaBox))
  
    deltalya[:] = 0
    if A.NmeshLyaBox > 1:
        # this is the kernel to correct for log normal transformation
        # as it matters when we get to near the JeansScale
        def kernel(kx, ky, kz, k):
            f2 = 1 / (1 + (A.JeansScale / (2 * numpy.pi) * k) ** 2)
            return f2
  
        cutoff = 0.5 * 2 * numpy.pi / A.BoxSize * A.NmeshEff
        seed = A.RNG.randint(1<<21 - 1, size=A.NLyaBox)
        def work(i):
            deltalya[i], varjunk = density.realize(A.power, 
                    seed[i],
                    A.NmeshLyaBox, 
                    A.BoxSize / A.NmeshEff, 
                    Kmin=cutoff,
                    kernel=kernel)
        with sharedmem.Pool() as pool:
            pool.map(work, range(A.NLyaBox))

    print 'lya field', 'mean', deltalya.mean(), 'var', deltalya.var()
    return deltalya.var()

class Visitor(object):
    M = {}
    @staticmethod
    def prepare(cls, A, Npixels, fields):
        """ prepare is called before the fork """
        cls.A = A
        for f in fields:
            if f not in Visitor.M: 
                Visitor.M[f] = A.P(f, memmap='w+', shape=Npixels)

    def __init__(self, box):
        """ init is called after the fork """
        self.box = box
        A = self.A

    def visit(self, chunk):
        pass

    def getcoarse(self, xyz):
        return xyz * (1.0 * self.A.NmeshCoarse / self.A.BoxSize)
    def getfine(self, xyz):
        return (xyz - self.box.origin[None, :]) * \
                (1.0 * self.A.NmeshFine / (self.A.BoxSize / self.A.Nrep))

    def getpixels(self, chunk, return_full):
        """ return_full is True, return xyzlya, objectid 
            False, return total number of pixels in chunk """
        s = chunk.getslice()
        result = \
            drawline(
                self.A.sightlines.x1[s], self.A.sightlines.x2[s], 
                self.A.JeansScale,
                self.box.origin, 
                self.box.origin + self.A.BoxSize / self.A.Nrep,
                return_full=return_full)
        if return_full:
            xyz, objectid, junk = result
            objectid += chunk.offset
            return xyz, objectid
        else:
            return result.sum()

class GeneratePixels(Visitor):
    @classmethod
    def prepare(cls, A, Npixels):
        Visitor.prepare(cls, A, Npixels, [
            'objectid', 
            'dc']
            )
        cls.MemoryBytes = 1

    def __init__(self, box, gaussian_save):
        Visitor.__init__(self, box)
    def visit(self, chunk):
        ps = slice(chunk.PixelStart, chunk.PixelEnd)
        xyz, objectid = self.getpixels(chunk, return_full=True)
        self.M['objectid'][ps] = objectid
        self.M['dc'][ps] = self.A.xyz2dc(xyz)

class AddDelta(Visitor):
    @classmethod
    def prepare(cls, A, Npixels):
        Visitor.prepare(cls, A, Npixels, [
            'delta']
            )
        cls.MemoryBytes = A.NmeshFine ** 3 * 8 * 2

    def __init__(self, box, gaussian_save):
        Visitor.__init__(self, box)
        A = self.A
        gaussian[...] = gaussian_save
        delta1, junk = \
            density.realize(A.power, 
               None, 
               A.NmeshFine, 
               A.BoxSize / A.Nrep, 
               Kmin=A.KSplit,
               gaussian=gaussian)
        self.var1 = delta1.var()
        self.delta1 = spline_filter(delta1, order=4, 
                output=numpy.dtype('f4'))
        self.RNG = numpy.random.RandomState(
               A.SeedTable[box.i, box.j, box.k] 
            )

    def visit(self, chunk):
        ps = slice(chunk.PixelStart, chunk.PixelEnd)
        xyz, objectid = self.getpixels(chunk, return_full=True)
        xyzcoarse = self.getcoarse(xyz)
        d = map_coordinates(delta0, xyzcoarse.T, 
                mode='wrap', order=4, prefilter=False)

        xyzfine = self.getfine(xyz)
        d[:] += map_coordinates(self.delta1, xyzfine.T, 
                mode='wrap', order=4, prefilter=False)
        N = len(xyzfine)
        d[:] += self.RNG.normal(scale=varlya ** 0.5, size=N)
        self.M['delta'][ps] = d


class Estimator(Visitor):
    @classmethod
    def prepare(cls, A):
        Visitor.prepare(cls, A, 0, [])

    def visit(self, chunk):
        chunk.Npixels[...] = self.getpixels(chunk, return_full=False)

def AddDisp(d):
  class AddDisp(Visitor):
    @classmethod
    def prepare(cls, A, Npixels):
        Visitor.prepare(cls, A, Npixels, [
            'losdisp']
            )
        cls.MemoryBytes = A.NmeshFine ** 3 * 8 * 2
    def __init__(self, box, gaussian_save):
        Visitor.__init__(self, box)
        A = self.A
        self.dir = A.sightlines.dir
        gaussian[...] = gaussian_save
        self.disp1, junk = \
            density.realize(A.power,
                None, 
                A.NmeshFine,
                A.BoxSize / A.Nrep, disp=d, Kmin=A.KSplit,
                gaussian=gaussian
                )
        self.disp1 = spline_filter(self.disp1, 
                order=4, output=numpy.dtype('f4'))
    def visit(self, chunk):
        ps = slice(chunk.PixelStart, chunk.PixelEnd)
        xyz, objectid = self.getpixels(chunk, return_full=True)
        xyzcoarse = self.getcoarse(xyz)
        xyzfine = self.getfine(xyz)
        proj = self.dir[objectid, d]
        disp = map_coordinates(disp0[d], 
                xyzcoarse.T,
                mode='wrap', order=4, prefilter=False)
        disp += map_coordinates(self.disp1, 
                xyzfine.T,
                mode='wrap', order=4, prefilter=False)
        self.M['losdisp'][ps] += disp * proj
  return AddDisp

