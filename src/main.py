import numpy
from args import parseargs, loadpixel, pixeldtype, pixeldtype2
from scipy.ndimage import map_coordinates
from scipy.optimize import fmin, leastsq
from density import realize, lognormal
import density
from splat import splat
from cosmology import interp1d
import sharedmem
from bresenham import clipline
#sharedmem.set_debug(True)

def main():
  A = parseargs()
  if A.serial:
      sharedmem.set_debug(True)

  if A.mode == 'firstpass':
    firstpass(A)
  if A.mode == 'secondpass':
    secondpass(A)
  if A.mode == 'thirdpass':
    thirdpass(A)
  if A.mode == 'fourthpass':
    fourthpass(A)
  if A.mode == 'fifthpass':
    fifthpass(A)

def firstpass(A):
  """ raw guassian field"""

  seed0 = numpy.random.randint(1<<21 - 1)
  seedtable = numpy.random.randint(1<<21 - 1, size=(A.Nrep,) * 3)

  if A.Geometry == 'Sphere':
      realize_losdisp = density.realize_dispr
  else:
      realize_losdisp = density.realize_dispz

  delta0 = realize(A.power, seed0, 
                 A.Nmesh, A.Nmesh, A.BoxSize)
  losdisp0 = realize_losdisp(A.power, [0, 0, 0],
               seed0, A.Nmesh, A.Nmesh, A.BoxSize)

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
    deltalya[i] = realize(A.power, seed,
            A.NmeshLya, A.NmeshLya, 
             A.BoxSize / A.NmeshEff, kernel=kernel)

  def work(i, j, k):
    x1 = A.sightlines.x1 / A.BoxSize * (A.NmeshLya * A.NmeshEff)
    x2 = A.sightlines.x2 / A.BoxSize * (A.NmeshLya * A.NmeshEff)
    offset = numpy.array([i, j, k]) * A.Nmesh * A.NmeshLya
    x1 -= offset
    x2 -= offset
  
    xyz, lineid, pixelid = clipline(x1, x2, (A.Nmesh * A.NmeshLya, ) * 3, return_lineid=True)
    print xyz.size, lineid.size, pixelid.size
    pixels = numpy.empty(lineid.size, dtype=pixeldtype)
    if len(pixels) == 0: return numpy.array([])
  
    pixels['row'] = lineid
    pixels['ind'] = pixelid
    coarse = (xyz + offset[:, None]) / (1.0 * A.NmeshLya * A.NmeshEff / A.Nmesh)
  
    assert (coarse < A.Nmesh).all()
    assert (coarse >= 0).all()
  
    pixels['delta'] = map_coordinates(delta0, coarse, mode='wrap', order=5)
    pixels['losdisp'] = map_coordinates(losdisp0, coarse, mode='wrap', order=3)
  
    if A.Geometry == 'Sphere':
        pixels['Z'] = ((coarse - 0.5 * A.Nmesh) ** 2).sum(axis=-1) ** 0.5
    else:
        pixels['Z'] = coarse[2]
        # offset the redshift
        pixels['Z'] += A.cosmology.Dc(1 / (A.Redshift + 1)) * (A.DH / A.BoxSize * A.Nmesh)

    pixels['Z'] *= (A.BoxSize / A.Nmesh) / A.DH
    pixels['Z'] = 1 / A.cosmology.aback(pixels['Z']) - 1
  
    if A.Nrep > 1:
      # add in the small scale power
      delta1 = realize(A.power, seedtable[i, j, k], A.NmeshEff, 
                 A.Nmesh, A.BoxSize, A.Nmesh)
      losdisp1 = realize_losdisp(A.power, offset / A.NmeshLya / A.NmeshEff * A.BoxSize,
                 seedtable[i, j, k], A.NmeshEff,
                 A.Nmesh, A.BoxSize, A.Nmesh)
      rng = numpy.random.RandomState(seedtable[i, j, k])
      Lyatable = rng.randint(A.NLyaBox, size=A.Nmesh ** 3)
  
      # add in the lya scale power
      fine = xyz / (1.0 * A.NmeshLya)
      assert (fine < A.Nmesh).all()
      assert (fine >= 0).all()
      pixels['delta'] += map_coordinates(delta1, fine, mode='wrap', order=3)
      pixels['losdisp'] += map_coordinates(losdisp1, fine, mode='wrap', order=3)
  
      fine = numpy.ravel_multi_index(numpy.int32(fine), (A.Nmesh, ) * 3)
      whichLyaBox = Lyatable[fine]
      lyaoffset = xyz % A.NmeshLya
      ind = numpy.ravel_multi_index((whichLyaBox, lyaoffset[0], lyaoffset[1],
          lyaoffset[2]), (A.NLyaBox, A.NmeshLya, A.NmeshLya, A.NmeshLya),
          mode='wrap')
      pixels['delta'] += deltalya.flat[ind]

    writefill(pixels, A.basename(i, j, k, 'pass1'), stat='delta')
    
  with sharedmem.Pool() as pool:
    pool.starmap(work, list(A.yieldwork()))


def secondpass(A):
  """ log normal transformation and redshift distortion"""
  mean, std = normalization(A)
  Dplus = A.cosmology.Dplus
  FOmega = A.cosmology.FOmega
  Ea = A.cosmology.Ea
  H0 = 0.1
  Dplus0 = Dplus(1.0)
  def work(i, j, k):
    fill1 = numpy.fromfile(A.basename(i, j, k, 'pass1') + '.raw',
            pixeldtype)
    fill2 = fill1.view(dtype=pixeldtype2)
    a = 1 / (1 + fill1['Z'])
    D = Dplus(a) / Dplus0
    fill1['delta'] = (fill1['delta'] - mean) * D
    lognormal(fill1['delta'], std=std * D, out=fill1['delta'])
    fill2['losvel'] *= D * FOmega(a) * Ea(a) * H0 * fill1['losdisp']
    # moving away is positive
    dz = - fill2['losvel'] / A.C
    fill2['Z'] = (1 + dz) * (1 + fill2['Z']) - 1
    fill2['F'] = numpy.exp(-numpy.exp(A.beta * fill1['delta']))
    writefill(fill2, A.basename(i, j, k, 'pass2')) 

  with sharedmem.Pool() as pool:
    pool.starmap(work, list(A.yieldwork()))

def thirdpass(A):
  """find the normalization factor matching the mean flux"""
  DH = A.C / A.H
  zbins = numpy.linspace(A.Zmin, A.Zmax, 100)
  zcenter = 0.5 * (zbins[1:] + zbins[:-1])
  afactor = numpy.ones_like(zcenter)
  meanflux_expected = A.FPGAmeanflux(1 / (1 + zcenter))
  print 'zmax = ', A.Zmax
  for i in range(len(zcenter)):
    all = []
    def work(ii, j, k):
      fill = numpy.memmap(A.basename(ii, j, k, 'pass2') +
              '.raw', mode='r', dtype=pixeldtype2)
      dig = numpy.digitize(fill['Z'], zbins)
      return fill['F'][dig == i + 1]
    def reduce(res):
      all.append(res)
    with sharedmem.Pool() as pool:
      pool.starmap(work, list(A.yieldwork()), callback=reduce)
    F = numpy.concatenate(all)
    if F.size > 0:
      with sharedmem.Pool(use_threads=True) as pool:
        def cost(az):
          def work(F):
            return (F[F!=0] ** az[0]).sum(dtype='f8')
          Fsum = numpy.sum(pool.starmap(work,
              pool.zipsplit((F,))))
          Fmean = Fsum / F.size
          dist = (Fmean - meanflux_expected[i])
          return dist
        if i > 1:
          p0 = afactor[i - 1]
        else:
          p0 = 1.0
        p = leastsq(cost, p0, full_output=False)
        afactor[i] = p[0]
    else:
      afactor[i] = 1.0
    print i, afactor[i], F.size
  numpy.savetxt(A.datadir + '/afactor.txt', zip(zcenter, afactor),
          fmt='%g')

def fourthpass(A):
  """ normalize to observed mean flux """
  zcenter, afactor = numpy.loadtxt(A.datadir + '/afactor.txt', unpack=True)
  AZ = interp1d(zcenter, afactor, fill_value=1.0, kind=4)

  def work(i, j, k):
    fill = numpy.fromfile(A.basename(i, j, k, 'pass2') + '.raw',
            pixeldtype2)
    fill['F'] = fill['F'] ** AZ(fill['Z'])
    writefill(fill, A.basename(i, j, k, 'pass4')) 

  with sharedmem.Pool() as pool:
    pool.starmap(work, list(A.yieldwork()))

def fifthpass(A):
  """rebin to observed sight lines, this going to take hack lot of memory"""
  bitmap = numpy.empty((A.sightlines.size, 51), 'f4')
  all = []
  for i, j, k in A.yieldwork():
      fill = numpy.memmap(A.basename(i, j, k, 'pass4') +
              '.raw', mode='r', dtype=pixeldtype2)
      all.append(fill) 

  all = numpy.concatenate(all)
  all.sort(order='row')
  start = all['row'].searchsorted(numpy.arange(len(bitmap)), side='left')
  end = all['row'].searchsorted(numpy.arange(len(bitmap)), side='right')
  for i in range(len(bitmap)):
      Z = all['Z'][start[i]:end[i]]
      F = all['F'][start[i]:end[i]]
      s = Z.argsort()
      Z = Z[s]
      F = F[s]
      bins = numpy.linspace(A.sightlines.Zmin[i], A.sightlines.Zmax[i], 50,
              endpoint=True)
      bitmap[i] = splat(Z, F, bins) / splat(Z, 1, bins)
  bitmap.tofile(A.datadir + '/bitmap.raw')
  
def normalization(A):
  K1sum = 0.0
  K2sum = 0.0
  Nsum = 0
  for ijk in range(A.Nrep ** 3):
    i, j, k = numpy.unravel_index(ijk, (A.Nrep, ) * 3)
    G = {}
    execfile(A.basename(i, j, k, 'pass1') + '.info-file', G)
    K1sum += G['K']
    K2sum += G['K2']
    Nsum += G['N']
  mean = K1sum / Nsum
  std = (K2sum / Nsum - mean ** 2) ** 0.5
  print 'mean is', mean, 'std is', std
  return mean, std

def writefill(fill, basename, stat=None):
    fill.tofile(basename + '.raw')
    if stat is not None:
      K2 = (fill['delta'] ** 2).sum(dtype='f8')
      K = fill['delta'].sum(dtype='f8')
      N = fill['delta'].size
      file(basename + '.info-file', mode='w').write(
"""
K2 = %(K2)g
K = %(K)g
N = %(N)g
""" % locals())
    print 'wrote', basename
main()

