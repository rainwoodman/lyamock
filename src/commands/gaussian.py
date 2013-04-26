import numpy
import density
from args import pixeldtype0, pixeldtype1, writefill
from bresenham import clipline
from scipy.ndimage import map_coordinates
import sharedmem

def main(A):
  """ raw guassian field"""

  seed0 = numpy.random.randint(1<<21 - 1)
  seedtable = numpy.random.randint(1<<21 - 1, size=(A.Nrep,) * 3)

  if A.Geometry == 'Test':
      realize_losdisp = density.realize_dispz
  elif A.Geometry == 'Sphere':
      realize_losdisp = density.realize_dispr
  else:
      raise Exception("Geometry unknown")

  print 'preparing large scale modes'

  delta0 = density.realize(A.power, seed0, 
                 A.Nmesh, A.Nmesh, A.BoxSize, order=3)
  losdisp0 = realize_losdisp(A.power, [0, 0, 0],
               seed0, A.Nmesh, A.Nmesh, A.BoxSize, order=3)
  print 'preparing LyaBoxes'
  # fill the lya resolution small boxes
  deltalya = numpy.empty((A.NLyaBox, A.NmeshLya, A.NmeshLya,
      A.NmeshLya))
  K0 = 2 * numpy.pi / (A.BoxSize / A.NmeshEff)

  # this is the kernel to correct for log normal transformation
  # as it matters when we get to near the JeansScale
  def kernel(kx, ky, kz, k):
     f1 = 1 # (1 + 0.4 * numpy.exp(- (k / K0 - 1) / 0.05))
     f2 = 1 / (1 + (A.JeansScale / (2 * numpy.pi) * k) ** 2)
     return f1 * f2

  for i in range(A.NLyaBox):
    seed = numpy.random.randint(1<<21 - 1)
    if A.NmeshLya > 1:
      deltalya[i] = density.realize(A.power, seed,
            A.NmeshLya, A.NmeshLya, 
             A.BoxSize / A.NmeshEff, kernel=kernel)

  print 'spawn and work on intermediate scales'
  def work(i, j, k):
    x1 = A.sightlines.x1 / A.BoxSize * (A.NmeshLya * A.NmeshEff)
    x2 = A.sightlines.x2 / A.BoxSize * (A.NmeshLya * A.NmeshEff)
    offset = numpy.array([i, j, k]) * A.Nmesh * A.NmeshLya
    # relative to the corner
    x1 -= offset
    x2 -= offset
    # skip the bad lines
    mask = A.sightlines.Zmin < A.sightlines.Zmax
    xyz, lineid, pixelid = clipline(x1[mask], x2[mask], (A.Nmesh * A.NmeshLya, ) * 3, return_lineid=True)
    pixels = numpy.empty(lineid.size, dtype=pixeldtype0)
    if len(pixels) == 0: 
        return numpy.array([])
    # recover the original lineid as it has been masked
    pixels['row'] = numpy.arange(len(A.sightlines))[mask][lineid]
    pixels['ind'] = pixelid

    del lineid
    del pixelid
    del mask
    del x1, x2

    pixels['delta'] = 0
    pixels['losdisp'] = 0

    def blend_pixels0(pixels, xyz):
        """ this will blend in the secondary mesh """
        coarse = (xyz + offset[:, None]) / (1.0 * A.NmeshLya *
             A.NmeshEff / A.Nmesh)
        pixels['delta'] += map_coordinates(delta0, coarse, 
                mode='wrap', order=3, prefilter=False)
        pixels['losdisp'] += map_coordinates(losdisp0, coarse, 
                mode='wrap', order=3, prefilter=False)
        if A.Geometry == 'Test':
            pixels['Z'] = coarse[2]
            # offset by the redshift
            pixels['Z'] += A.cosmology.Dc(1 / (A.Redshift + 1)) \
                    * (A.DH / A.BoxSize * A.Nmesh)
        elif A.Geometry == 'Sphere':
            pixels['Z'] = ((coarse - 0.5 * A.Nmesh) ** 2).sum(axis=0) ** 0.5
        else:
            raise Exception('Geometry unknown')

        pixels['Z'] *= (A.BoxSize / A.Nmesh) / A.DH
        pixels['Z'] = 1 / A.cosmology.aback(pixels['Z']) - 1

    for ch in range(0, len(xyz.T), 4096):
        SL = slice(ch, ch + 4096)
        blend_pixels0(pixels[SL], xyz[:, SL])
  
    if A.Nrep > 1:
      # add in the small scale power
      delta1 = density.realize(A.power, seedtable[i, j, k], A.NmeshEff, 
                 A.Nmesh, A.BoxSize, A.Nmesh, order=3)
      losdisp1 = realize_losdisp(A.power, offset / A.NmeshLya / A.NmeshEff * A.BoxSize,
                 seedtable[i, j, k], A.NmeshEff,
                 A.Nmesh, A.BoxSize, A.Nmesh, order=3)

      rng = numpy.random.RandomState(seedtable[i, j, k])
      Lyatable = rng.randint(A.NLyaBox, size=A.Nmesh ** 3)
  
      def blend_pixels1(pixels, xyz):
          """ this will blend in the lyman alpha mesh """
          fine = xyz / (1.0 * A.NmeshLya)
          pixels['delta'] += map_coordinates(delta1, fine, 
                  mode='wrap', order=3, prefilter=False)
          pixels['losdisp'] += map_coordinates(losdisp1, fine, 
                  mode='wrap', order=3, prefilter=False)
          if A.NmeshLya > 1:
            # add in the lya scale power
            fine = numpy.ravel_multi_index(numpy.int32(fine), (A.Nmesh, ) * 3)
            whichLyaBox = Lyatable[fine]
            lyaoffset = xyz % A.NmeshLya
            ind = numpy.ravel_multi_index((whichLyaBox, lyaoffset[0], lyaoffset[1],
                lyaoffset[2]), (A.NLyaBox, A.NmeshLya, A.NmeshLya, A.NmeshLya),
                mode='wrap')
            pixels['delta'] += deltalya.flat[ind]
      for ch in range(0, len(xyz.T), 4096):
          SL = slice(ch, ch + 4096)
          blend_pixels1(pixels[SL], xyz[:, SL])
      
    # actually, we are writing out pixeldtype1, losdisp saves Zshift
    pixels['losdisp'] = Zdistort(A, pixels['Z'], pixels['losdisp'])

    writefill(pixels, A.basename(i, j, k, 'pass1'), stat='delta')
    
  with sharedmem.Pool() as pool:
    pool.starmap(work, list(A.yieldwork()))

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
