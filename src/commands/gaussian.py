import numpy
from bresenham import clipline
import sharedmem
import density
from scipy.ndimage import spline_filter, map_coordinates

def main(A):
    var0 = initcoarse(A)
    var1 = initfine1(A)
    varlya = initlya(A)
    print 'gaussian-variance is', var0 + var1 + varlya
    numpy.savetxt(A.datadir + '/gaussian-variance.txt', [var0 + var1 + varlya])
    
    layout = A.layout(len(A.sightlines), chunksize=128)

    layout.Npixels = sharedmem.empty((A.Nrep,) * 3,
            dtype=('intp', (layout.Nchunks, )))

    Estimator.prepare(A)
    with sharedmem.Pool() as pool:
        def work(i, j, k):
            box = layout[i, j, k]
            proc = Estimator(box)
            for chunk in box:
                proc.visit(chunk)
        pool.starmap(work, A.yieldwork())

    layout.PixelEnd = layout.Npixels.cumsum().reshape(layout.Npixels.shape)
    layout.PixelStart = layout.PixelEnd.copy()
    layout.PixelStart[0] = 0
    layout.PixelStart.flat[1:] = layout.PixelEnd.flat[:-1]

    processors = [ 
            GeneratePixels,
            AddDelta,
            AddLya,
            AddDisp(0, 'dispx'),
            AddDisp(1, 'dispy'),
            AddDisp(2, 'dispz'),
        ]

    M = {}
    for proc in processors:
        proc.prepare(A, layout.Npixels.sum())

    print 'spawn and work'
    with sharedmem.Pool() as pool:
        def work(i, j, k):
            box = layout[i, j, k]
            if box.Npixels.sum() == 0: return
            for cls in processors:
                proc = cls(box)
                for chunk in box:
                    proc.visit(chunk)
                # free memory
                del proc 
            print 'done', i, j, k
        pool.starmap(work, A.yieldwork())

    for field in Visitor.M:
        Visitor.M[field].flush()

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

def initfine1(A):
    junk, var1 = density.realize(A.power, 
               A.SeedTable[0, 0, 0], 
               A.NmeshFine, 
               A.BoxSize / A.Nrep, 
               Kmin=A.KSplit)
    return var1

def initlya(A):
    global deltalya
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
        cls.A = A
        for f in fields:
            if not f in Visitor.M:
                try:
                    Visitor.M[f] = A.P(f, memmap='r+', shape=Npixels)
                except IOError:
                    Visitor.M[f] = A.P(f, memmap='w+', shape=Npixels)
            cls.M[f] = Visitor.M[f]

    def __init__(self, box):
        self.box = box
        A = self.A
        fac = A.NmeshLyaBox * A.NmeshFine * A.Nrep / A.BoxSize
        self.x1lya = A.sightlines.x1 * fac
        self.x2lya = A.sightlines.x2 * fac

    def visit(self, chunk):
        pass

    def getcoarse(self, xyzlya):
        return 1.0 * xyzlya * self.A.NmeshCoarse / (self.A.NmeshFine * self.A.NmeshLyaBox)
    def getfine(self, xyzlya):
        return (xyzlya - self.box.LYAoffset[None, :]) \
                / self.A.NmeshLyaBox
    def getqso(self, xyzlya):
        return 1.0 * (xyzlya - self.box.LYAoffset[None, :]) \
                / self.A.NmeshLyaBox * self.A.NmeshQSO / self.A.NmeshFine

    def getpixels(self, chunk, return_full):
        """ return_full is True, return xyzlya, objectid, pixelid
            False, return total number of pixels in chunk """
        s = chunk.getslice()
        result = \
            clipline(
                self.x1lya[s], self.x2lya[s], 
                self.box.LYAoffset, 
                self.box.LYAoffset + self.box.LYAsize,
                return_full=return_full)
        if return_full:
            xyzlya, objectid, pixelid = result
            objectid += chunk.offset
            return xyzlya, objectid, pixelid
        else:
            return result.sum()

class GeneratePixels(Visitor):
    @classmethod
    def prepare(cls, A, Npixels):
        Visitor.prepare(cls, A, Npixels, [
            'objectid', 
            'pixelid', 
            'Zreal', ]
            )

    def visit(self, chunk):
        ps = slice(chunk.PixelStart, chunk.PixelEnd)
        xyzlya, objectid, pixelid = self.getpixels(chunk, return_full=True)
        self.M['objectid'][ps] = objectid
        self.M['pixelid'][ps] = pixelid
        pos = self.A.BoxSize * xyzlya / (self.A.NmeshEff * self.A.NmeshLyaBox)
        self.M['Zreal'][ps] = self.A.xyz2redshift(pos)

class AddDelta(Visitor):
    @classmethod
    def prepare(cls, A, Npixels):
        Visitor.prepare(cls, A, Npixels, [
            'delta']
            )

    def __init__(self, box):
        Visitor.__init__(self, box)
        A = self.A
        self.delta1, junk = \
            density.realize(A.power, 
               A.SeedTable[box.i, box.j, box.k], 
               A.NmeshFine, 
               A.BoxSize / A.Nrep, 
               Kmin=A.KSplit)

    def visit(self, chunk):
        ps = slice(chunk.PixelStart, chunk.PixelEnd)
        xyzlya, objectid, pixelid = self.getpixels(chunk, return_full=True)
        xyzcoarse = self.getcoarse(xyzlya)
        self.M['delta'][ps] = map_coordinates(delta0, xyzcoarse.T, 
                mode='wrap', order=4, prefilter=False)

        xyzfine = self.getfine(xyzlya)
        linear = numpy.ravel_multi_index(xyzfine.T, 
                (self.A.NmeshFine, ) * 3, mode='raise')
        self.M['delta'][ps] += self.delta1.flat[linear]

class AddLya(Visitor):
    @classmethod
    def prepare(cls, A, Npixels):
        Visitor.prepare(cls, A, Npixels, [
            'delta']
            )
    def __init__(self, box):
        Visitor.__init__(self, box)
        A = self.A
        RNG = numpy.random.RandomState(
               A.SeedTable[box.i, box.j, box.k] 
            )
        self.Lyatable = numpy.empty(A.NmeshFine ** 3, dtype='i2')
        for i in range(A.NmeshFine):
            self.Lyatable[i * A.NmeshFine ** 2:(i+1) * A.NmeshFine ** 2] \
                = RNG.randint(A.NLyaBox, size=A.NmeshFine ** 2)
    def visit(self, chunk):
        ps = slice(chunk.PixelStart, chunk.PixelEnd)
        xyzlya, objectid, pixelid = self.getpixels(chunk, return_full=True)
        xyzfine = self.getfine(xyzlya)
        linear = numpy.ravel_multi_index(xyzfine.T, 
                (self.A.NmeshFine, ) * 3, mode='raise')
        whichLyaBox = self.Lyatable[linear]
        ind = numpy.ravel_multi_index(
                (whichLyaBox, xyzlya[..., 0], xyzlya[..., 1], xyzlya[..., 2]), 
                deltalya.shape, mode='wrap')
        self.M['delta'][ps] += deltalya.flat[ind]

class Estimator(Visitor):
    @classmethod
    def prepare(cls, A):
        Visitor.prepare(cls, A, 0, [])

    def visit(self, chunk):
        chunk.Npixels[...] = self.getpixels(chunk, return_full=False)

def AddDisp(d, field):
  class AddDisp(Visitor):
    @classmethod
    def prepare(cls, A, Npixels):
        Visitor.prepare(cls, A, Npixels, [
            field]
            )
    def __init__(self, box):
        Visitor.__init__(self, box)
        A = self.A
        self.disp1, junk = \
            density.realize(A.power,
                A.SeedTable[box.i, box.j, box.k], 
                A.NmeshQSO,
                A.BoxSize / A.Nrep, disp=d, Kmin=A.KSplit
                )
        self.disp1 = spline_filter(self.disp1, 
                order=4, output=numpy.dtype('f4'))
    def visit(self, chunk):
        ps = slice(chunk.PixelStart, chunk.PixelEnd)
        xyzlya, objectid, pixelid = self.getpixels(chunk, return_full=True)
        xyzcoarse = self.getcoarse(xyzlya)
        xyzqso = self.getqso(xyzlya)

        self.M[field][ps] = map_coordinates(disp0[d], 
                xyzcoarse.T,
                mode='wrap', order=4, prefilter=False)

        self.M[field][ps] += map_coordinates(self.disp1, 
                xyzqso.T,
                mode='wrap', order=4, prefilter=False)
  return AddDisp

