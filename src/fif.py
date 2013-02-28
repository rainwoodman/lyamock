import numpy
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import map_coordinates
import sharedmem
from sys import argv

pixeldtype = numpy.dtype(
    [ ('pos', ('f4', 3)), ('row', 'i4'), ('ind', 'i4'), ('Z', 'f4')])

def loadpower(path=None):
  if path is None:
    import pycamb
    from gaepsi.cosmology import WMAP7
    args = dict(H0=WMAP7.h * 100, 
          omegac=WMAP7.M - 0.044, 
          omegab=0.044, 
          omegav=WMAP7.L, 
          omegak=0,
          omegan=0)
    fakesigma8 = pycamb.transfers(scalar_amp=1, **args)[2]
    sigma8 = 0.8
    scalar_amp = fakesigma8 ** -2 * sigma8 ** 2
    k, p = pycamb.matter_power(scalar_amp=scalar_amp, maxk=10, **args)
    k /= 1000
    p *= 1e9 * (2 * numpy.pi) ** -3 # xiao to GADGET
    print 'sigma8', pycamb.transfers(scalar_amp=scalar_amp, **args)[2]
  else:
    k, p = numpy.loadtxt(path, unpack=True)
  p[numpy.isnan(p)] = 0
  return interp1d(k, p, kind='linear', copy=True, bounds_error=False, fill_value=0)

def loadpixel(i, j, k, prefix='32/'):
  try:
    return numpy.fromfile(prefix + "%02d-%02d-%02d-pixels.raw" % (i, j, k), dtype=pixeldtype)
  except IOError:
    return numpy.array([], dtype=pixeldtype)

def realize(PowerSpec,
            seed, Nmesh, Nsample, BoxSize, 
            NmeshCoarse=None,
            kernel=lambda kx, ky, kz, k: 1):
  """
      realize a powerspectrum.
      PowerSpec must be in gadget unit, (2 pi kpc/h) ** 3

      default is to make delta field.

      to get displacement field, use
      lambda kx, ky, kz, k: I * kx * k ** -2

      The actual boxsize is BoxSize / Nmesh * Nsample.
      The actual resolution is BoxSize / Nmesh.

      if NmeshCoarse is not None, leave out a region of
      size NmeshCoarse * K0 in the middle of the power.
      presumbaly power from region will be added as large
      scale power from a different realization with interpolation.
  """
  rng = numpy.random.RandomState(seed)
  K0 = 2 * numpy.pi / (BoxSize * Nsample / Nmesh)
  gauss = rng.normal(scale=2 ** -0.5, 
            size=(Nsample, Nsample, Nsample / 2 + 1, 2)
          ).view(dtype=numpy.complex128).reshape(Nsample, Nsample, Nsample / 2 + 1)

  # convolve gauss with powerspec to delta_k, stored in gauss.
  for i in range(Nsample):
    # loop over i to save memory
    if i > Nsample / 2: i = i - Nsample
    j, k = numpy.ogrid[0:Nsample, 0:Nsample/2+1]
    j[j > Nsample / 2] = j[j > Nsample / 2] - Nsample
    K = numpy.empty((Nsample, Nsample/2 + 1), dtype='f4')
    K[...] = (i ** 2 + j ** 2 + k ** 2) ** 0.5 * K0
    PK = PowerSpec(K) ** 0.5
    gauss[i] *= (PK * K0 ** 1.5) * kernel(i * K0, j * K0, k * K0, K)
    gauss[i][numpy.isnan(gauss[i])] = 0
    if NmeshCoarse:
      thresh = NmeshCoarse * Nsample / Nmesh / 2
      if i < thresh and i > - thresh:
        mask = (j < thresh) & (k < thresh) & \
               (j > - thresh) & (k > -thresh)
        gauss[i][mask] = 0
  
  gauss[Nsample / 2, ...] = 0
  gauss[:, Nsample / 2, :] = 0
  gauss[:, :, Nsample / 2] = 0
  delta = numpy.fft.irfftn(gauss)
  delta *= Nsample ** 3
  return numpy.float32(delta)

def realize_losdisp(PowerSpec, cornerpos,
            seed, Nmesh, Nsample, BoxSize, 
            NmeshCoarse=None):
  dispkernel = [
    lambda kx, ky, kz, k: 1j * kx * k ** -2,
    lambda kx, ky, kz, k: 1j * ky * k ** -2,
    lambda kx, ky, kz, k: 1j * kz * k ** -2]
  losdisp = numpy.zeros((Nsample, Nsample, Nsample), dtype='f4')
  for ax in range(3):
    disp = realize(PowerSpec, seed=seed, Nmesh=Nmesh, 
           Nsample=Nsample, BoxSize=BoxSize, NmeshCoarse=NmeshCoarse,
           kernel=dispkernel[ax])
    for i in range(Nsample):
      j, k = numpy.ogrid[0:Nsample, 0:Nsample]
      x = cornerpos[0] + i * (BoxSize / Nmesh * Nsample) - BoxSize * 0.5
      y = cornerpos[1] + j * (BoxSize / Nmesh * Nsample) - BoxSize * 0.5
      z = cornerpos[2] + k * (BoxSize / Nmesh * Nsample) - BoxSize * 0.5
      disp[i] *= ([-x, -y, -z][ax])
      disp[i] *= (x ** 2 + y ** 2 + z ** 2) ** -0.5
    losdisp += disp
  return losdisp

numpy.random.seed(1811720)
Box = 12000000
#PowerSpec = loadpower('power-16384.txt')
PowerSpec = loadpower()
NmeshEff = 8192
Nsample = 256 

if len(argv) == 1: raise Exception("will not run")

Nrep = NmeshEff / Nsample
seed0 = numpy.random.randint(1<<21 - 1)
seedtable = numpy.random.randint(1<<21 - 1, size=(Nrep, Nrep, Nrep))
delta0 = realize(PowerSpec, seed0, Nsample, Nsample, Box)
losdisp0 = realize_losdisp(PowerSpec, [0, 0, 0],
               seed0, Nsample, Nsample, Box)

def dopixels(i, j, k):
  pixels = loadpixel(i, j, k, prefix='%d/' % Nrep)
  if len(pixels) == 0: return numpy.array([])
  cornerpos = [i * Box / Nrep, j * Box / Nrep, k * Box / Nrep]
  delta1 = realize(PowerSpec, seedtable[i, j, k], NmeshEff, 
               Nsample, Box, Nsample)
  losdisp1 = realize_losdisp(PowerSpec, cornerpos,
               seedtable[i, j, k], NmeshEff,
               Nsample, Box, Nsample)
  coarse = pixels['pos'] / Box * Nsample
  fine = (pixels['pos'] / Box * Nrep - [i, j, k]) * Nsample
  d0 = map_coordinates(delta0, coarse.T, mode='wrap')
  disp0 = map_coordinates(losdisp0, coarse.T, mode='wrap')
  d1 = map_coordinates(delta1, fine.T, mode='wrap')
  disp1 = map_coordinates(losdisp1, fine.T, mode='wrap')
  return numpy.array([d1 + d0, disp0 + disp1]).T

if __name__ == '__main__':
  i = int(argv[1])
  def work(jk):
    j = jk // Nrep
    k = jk % Nrep
    delta = dopixels(i, j, k)
    if len(delta) == 0: return
    print i, j, k, delta.shape
    delta.tofile('out-%d/%02d-%02d-%02d-delta.raw' % (Nrep, i, j, k))
  
  with sharedmem.Pool() as pool:
    pool.map(work, range(Nrep * Nrep))
  
