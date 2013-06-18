import numpy
import density
from args import pixeldtype
from bresenham import clipline
from scipy.ndimage import map_coordinates
import sharedmem
import time
   
def main(A):
    """ raw guassian field"""
    global realize_losdisp
    global deltalya
    global delta0, losdisp0
    global F

    F= {}
    for field in pixeldtype.fields:
        F[field] = A.P(field, justfile='w')


    if A.Geometry == 'Test':
        realize_losdisp = density.realize_dispz
    elif A.Geometry == 'Sphere':
        realize_losdisp = density.realize_dispr
    else:
        raise Exception("Geometry unknown")
  
    print 'using', len(A.sightlines), 'sightlines'
    print 'preparing large scale modes'
  
    delta0, var0 = density.realize(A.power, 
                   A.Seed, 
                   A.NmeshCoarse, 
                   A.BoxSize, order=4,
                   Kmax=A.KSplit)
    losdisp0 = realize_losdisp(A.power, 
                 [0, 0, 0], A.Observer,
                 A.Seed, A.NmeshCoarse,
                 A.BoxSize, order=4,
                 Kmax=A.KSplit)
  
    print 'coarse field var', var0
    print 'preparing LyaBoxes'
  
    # fill the lya resolution small boxes
    deltalya = sharedmem.empty((A.NLyaBox, 
        A.NmeshLya, A.NmeshLya, A.NmeshLya))
  
    deltalya[:] = 0
    if A.NmeshLya > 1:
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
                    A.NmeshLya, 
                    A.BoxSize / A.NmeshEff, 
                    Kmin=cutoff,
                    kernel=kernel)
        with sharedmem.Pool(use_threads=False) as pool:
            pool.map(work, range(A.NLyaBox))

    print 'lya field', 'mean', deltalya.mean(), 'var', deltalya.var()
    print 'spawn and work on intermediate scales'

    time.clock()

    with sharedmem.Pool(use_threads=False) as pool:
        def work(i, j, k):
            var1, w1 = process_box(A, i, j, k, pool)
            return var1, w1
        rt = pool.starmap(work, A.yieldwork())
    var1 = numpy.average(
            [r[0] for r in rt],
            weights=[r[1] for r in rt])

    print 'fine field var', var1
    var = var0 + var1 + deltalya.var()
    print 'gaussian-variance is', var
    numpy.savetxt(A.datadir + '/gaussian-variance.txt', [var])

def process_box(A, i, j, k, pool):
    """ this will return number of pixels, variance of delta1, and w
    where w is 0 if no fft on delta1 is done, and 1 if fft is done.
    idea is we add up variance of delta0, delta1 and lya to get the
    full variance.
    """
    offset = numpy.array([i, j, k], dtype='f8')
    x1 = A.sightlines.x1 * (A.NmeshLyaEff / A.BoxSize) \
         - offset[None, :] * (A.NmeshLyaEff / A.Nrep)
    x2 = A.sightlines.x2 * (A.NmeshLyaEff / A.BoxSize) \
         - offset[None, :] * (A.NmeshLyaEff / A.Nrep)
    
    # xyz is in lya coordinate, relative to the current box
    t = time.clock()
    mask = clipline(x1, x2, 
                (A.NmeshLyaEff / A.Nrep,) * 3, 
                return_full=False) > 0
    #print i, j, k, 'clipping took', time.clock() - t
    mask &= A.sightlines.Zmax > A.sightlines.Zmin
    if not mask.any(): return 0, 0

    truelineid = mask.nonzero()[0]

    t = time.clock()

    losdisp1 = realize_losdisp(A.power, 
               offset * (A.BoxSize / A.Nrep),
               A.Observer,
               A.SeedTable[i, j, k], A.NmeshFine,
               A.BoxSize / A.Nrep, Kmin=A.KSplit, order=False)

    delta1, var1 = density.realize(A.power, 
           A.SeedTable[i, j, k], 
           A.NmeshFine, 
           A.BoxSize / A.Nrep, 
           Kmin=A.KSplit, order=False)

    RNG = numpy.random.RandomState(A.SeedTable[i, j, k])
    Lyatable = RNG.randint(A.NLyaBox, size=A.NmeshFine ** 3)
    #print i, j, k, 'fft took', time.clock() - t

    # doing 1024 lines a time
    for chunk in numpy.array_split(truelineid, 
               len(truelineid) / 5000 + 1):

        xyzlya, rellineid, pixelid = clipline(x1[chunk], x2[chunk], 
                    (A.NmeshLyaEff / A.Nrep,) * 3, 
                     return_lineid=True)
    
        assert len(rellineid) > 0

        # pixels contains the pixels in this box, of all lines.
        pixels = numpy.empty(len(rellineid), dtype=pixeldtype)

        t = time.clock()
        # recover the original lineid as it has been masked
        pixels['objectid'] = chunk[rellineid]
        pixels['pixelid'] = pixelid
    
        del [rellineid, pixelid]

        pixels['delta'] = 0
        pixels['losdisp'] = 0

        # first convert to the coarse coordinate
        # relative to the global box
        # so that it is ready for interpolation
        xyz = xyzlya * (1. * A.NmeshCoarse / A.NmeshLyaEff)
        xyz += offset[:, None] * A.NmeshCoarse / A.Nrep

        pixels['delta'] += map_coordinates(delta0, xyz, 
                mode='wrap', order=4, prefilter=False)
        pixels['losdisp'] += map_coordinates(losdisp0, xyz, 
                mode='wrap', order=4, prefilter=False)

        # now goto the boxsize coordinate
        # to evaluate the redshift
        xyz *= A.BoxSize / A.NmeshCoarse
        pixels['Zreal'] = A.xyz2redshift(xyz)
  
        # first convert to the fine mesh coordinate
        # relative to the fine boxes.
        xyz = xyzlya / A.NmeshLya
        # no need to remove the offset for interpolation
        # it's periodic. 
        #  xyz -= offset * A.NmeshFineEff / A.Nrep

        # add in the lya scale power
        linear = numpy.ravel_multi_index(xyz, 
                (A.NmeshFine, ) * 3, mode='wrap')
        pixels['delta'] += delta1.flat[linear]
        pixels['losdisp'] += losdisp1.flat[linear]
#        pixels['delta'] += map_coordinates(delta1, xyz, 
#                mode='wrap', order=4, prefilter=False)
#        pixels['losdisp'] += map_coordinates(losdisp1, xyz, 
#                mode='wrap', order=4, prefilter=False)

        whichLyaBox = Lyatable[linear]

        # no need to wrap, ravel_multi_index will do it
        ind = numpy.ravel_multi_index(
                (whichLyaBox, xyzlya[0], xyzlya[1], xyzlya[2]), 
                deltalya.shape, mode='wrap')
        pixels['delta'] += deltalya.flat[ind]

        pixels['Zred'] = Zred(A, pixels['Zreal'], pixels['losdisp'])
        #print i, j, k, 'interpolation', 'done', time.clock() - t
        with pool.lock:
            t = time.clock()
            print i, j, k, 'writing out', len(pixels)
            for field in ['delta', 'objectid', 'pixelid', 'losdisp',
                          'Zreal', 'Zred']:
                pixels[field].tofile(F[field])
                F[field].flush()
            print i, j, k, 'writing out', 'done', time.clock() - t
    return var1, 1

def Zred(A, Z0, losdisp):
    a = 1 / (1. + Z0)
    Ea = A.cosmology.Ea
    H0 = A.H0
    Dplus = A.cosmology.Dplus
    FOmega = A.cosmology.FOmega
    D = Dplus(a) / Dplus(1.0)
    losvel = a * D * FOmega(a) * Ea(a) * H0 * losdisp
    # moving away is positive
    dz = - losvel / A.C
    Zred = (1 + dz) * (1 + Z0) - 1
    return Zred
