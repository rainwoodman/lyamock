import numpy
import density
from args import pixeldtype0, pixeldtype1, writefill
from bresenham import clipline
from scipy.ndimage import map_coordinates
import sharedmem

def main(A):
    """ raw guassian field"""
    global realize_losdisp
    global deltalya

    if A.Geometry == 'Test':
        realize_losdisp = density.realize_dispz
    elif A.Geometry == 'Sphere':
        realize_losdisp = density.realize_dispr
    else:
        raise Exception("Geometry unknown")
  
    print 'using', len(A.sightlines), 'sightlines'
    print 'preparing large scale modes'
  
    delta0 = density.realize(A.power, 
                   A.Seed, 
                   A.NmeshCoarse, 
                   A.BoxSize, order=3)
    losdisp0 = realize_losdisp(A.power, 
                 [0, 0, 0], A.Observer,
                 A.Seed, A.NmeshCoarse,
                 A.BoxSize, order=3)
  
    print 'preparing LyaBoxes'
  
    # fill the lya resolution small boxes
    deltalya = numpy.zeros((A.NLyaBox, 
        A.NmeshLya, A.NmeshLya, A.NmeshLya))
  
    if A.NmeshLya > 1:
        # this is the kernel to correct for log normal transformation
        # as it matters when we get to near the JeansScale
        def kernel(kx, ky, kz, k):
            f2 = 1 / (1 + (A.JeansScale / (2 * numpy.pi) * k) ** 2)
            return f2
  
        cutoff = 0.5 * 2 * numpy.pi / A.BoxSize * A.NmeshEff
        for i in range(A.NLyaBox):
            seed = numpy.random.randint(1<<21 - 1)
            deltalya[i] = density.realize(A.power, 
                    seed,
                    A.NmeshLya, 
                    A.BoxSize / A.NmeshEff, 
                    CutOff=cutoff,
                    kernel=kernel)
  
    print 'spawn and work on intermediate scales'

    # skip the bad lines
    mask = A.sightlines.Zmin < A.sightlines.Zmax
    x1 = A.sightlines.x1[mask] / A.BoxSize * A.NmeshLyaEff
    x2 = A.sightlines.x2[mask] / A.BoxSize * A.NmeshLyaEff

    with sharedmem.Pool(use_threads=False) as pool:
        def work(i, j, k):
            process_box(A, i, j, k, 
                    delta0, losdisp0, 
                    x1, x2, mask.nonzero()[0])
        pool.starmap(work, list(A.yieldwork()))


def process_box(A, i, j, k, delta0, losdisp0,
        x1, x2, usedlines):
    offset = numpy.array([i, j, k], dtype='f8') / A.Nrep * A.NmeshLyaEff
    
    # xyz is in lya coordinate, relative to the current box
    xyz, lineid, pixelid = clipline(x1 - offset[None, :], x2 - offset[None, :], 
                (A.NmeshLyaEff / A.Nrep,) * 3, 
                return_lineid=True)

    if lineid.size == 0: 
        return numpy.array([])

    # pixels contains the pixels in this box, of all lines.
    pixels = numpy.empty(lineid.size, dtype=pixeldtype0)

    # recover the original lineid as it has been masked
    pixels['row'] = usedlines[lineid]
    pixels['ind'] = pixelid

    del [lineid, pixelid]

    pixels['delta'] = 0
    pixels['losdisp'] = 0

    def blend_pixels0(pixels, xyz):
        """ this will blend in the coarse mesh """
        # first convert to the coarse coordinate
        # relative to the global box
        xyz = (xyz + offset[:, None]) * \
                (1. * A.NmeshCoarse / A.NmeshLyaEff)

        pixels['delta'] += map_coordinates(delta0, xyz, 
                mode='wrap', order=3, prefilter=False)
        pixels['losdisp'] += map_coordinates(losdisp0, xyz, 
                mode='wrap', order=3, prefilter=False)

        # now goto the box coordinate
        xyz *= A.BoxSize / A.NmeshCoarse
        if A.Geometry == 'Test':
            R = xyz[2].copy()
            # offset by the redshift
            R += A.cosmology.Dc(1 / (A.Redshift + 1)) * A.DH
        elif A.Geometry == 'Sphere':
            R = ((xyz - A.Observer[:, None]) ** 2).sum(axis=0) ** 0.5
        else:
            raise Exception('Geometry unknown')

        R /= A.DH
        pixels['Z'] = 1 / A.cosmology.aback(R) - 1

    for ch in range(0, len(xyz.T), 4096):
        SL = slice(ch, ch + 4096)
        blend_pixels0(pixels[SL], xyz[:, SL])
  
    if A.Nrep > 1:
        process_box_fine(A, i, j, k, pixels, xyz)

    # actually, we are writing out pixeldtype1, losdisp saves Zshift
    pixels['losdisp'] = Zdistort(A, pixels['Z'], pixels['losdisp'])

    writefill(pixels, A.basename(i, j, k, 'pass1'), stat='delta')

def process_box_fine(A, i, j, k, pixels, xyz):

    # add in the small scale power
    cutoff = 0.5 * 2 * numpy.pi / A.BoxSize * A.NmeshCoarse
    delta1 = density.realize(A.power, 
               A.SeedTable[i, j, k], 
               A.NmeshFine, 
               A.BoxSize / A.Nrep, 
               CutOff=cutoff, order=3)
    offset = numpy.array([i, j, k], dtype='f8')[:, None] / A.Nrep * A.BoxSize

    losdisp1 = realize_losdisp(A.power, offset,
               A.Observer,
               A.SeedTable[i, j, k], A.NmeshFine,
               A.BoxSize / A.Nrep, CutOff=cutoff, order=3)

    rng = numpy.random.RandomState(A.SeedTable[i, j, k])
    Lyatable = rng.randint(A.NLyaBox, size=A.NmeshFine ** 3)
  
    def blend_pixels1(pixels, xyz):
        """ this will blend in the lyman alpha mesh """
        # first convert to the fine mesh coordinate
        # relative to the rep box.
        xyz = xyz / A.NmeshLya
        pixels['delta'] += map_coordinates(delta1, xyz, 
                mode='wrap', order=3, prefilter=False)
        pixels['losdisp'] += map_coordinates(losdisp1, xyz, 
                mode='wrap', order=3, prefilter=False)
        if A.NmeshLya == 1:
            return

        # add in the lya scale power
        linear = numpy.ravel_multi_index(numpy.int32(xyz), 
                (A.NmeshFine, ) * 3)
        whichLyaBox = Lyatable[linear]
        lyaoffset = xyz % A.NmeshLya
        ind = numpy.ravel_multi_index(
                (whichLyaBox, lyaoffset[0], lyaoffset[1], lyaoffset[2]), 
                deltalya.shape, mode='wrap')
        pixels['delta'] += deltalya.flat[ind]

    for ch in range(0, len(xyz.T), 4096):
          SL = slice(ch, ch + 4096)
          blend_pixels1(pixels[SL], xyz[:, SL])
      
        
def Zdistort(A, Z0, losdisp):
    a = 1 / (1. + Z0)
    Ea = A.cosmology.Ea
    H0 = A.H0
    Dplus = A.cosmology.Dplus
    FOmega = A.cosmology.FOmega
    D = Dplus(a) / Dplus(1.0)
    losvel = a * D * FOmega(a) * Ea(a) * H0 * losdisp
    print 'losvel stat', losvel.max(), losvel.min()
    # moving away is positive
    dz = - losvel / A.C
    Zshift = (1 + dz) * (1 + Z0) - 1 - Z0
    print 'Z0', Z0.max(), Z0.min()
    print 'Zshift', Zshift.max(), Zshift.min()
    return Zshift
