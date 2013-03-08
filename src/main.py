import numpy
from args import parseargs, loadpixel
from scipy.ndimage import map_coordinates
import sharedmem
def main():
  global A, delta0, losdisp0
  A = parseargs()
  loadpixel(A, 0, 1, 0)
  delta0 = realize(A.power, A.seed0, 
                 A.Nmesh, A.Nmesh, A.BoxSize)
  losdisp0 = realize_losdisp(A.power, [0, 0, 0],
               A.seed0, A.Nmesh, A.Nmesh, A.BoxSize)

  def work(jk):
    j = jk // A.Nrep
    k = jk % A.Nrep
    fill = fillpixels(A, delta0, losdisp0, A.i, j, k)
    fill.tofile('%s/%02d-%02d-%02d-delta.raw' % (A.datadir, A.i, j, k))
  with sharedmem.Pool() as pool:
    pool.map(work, range(A.Nrep * A.Nrep))

def fillpixels(A, delta0, losdisp0, i, j, k):
  pixels = loadpixel(A, i, j, k)
  if len(pixels) == 0: return numpy.array([])
  RepSpacing = A.BoxSize / A.Nrep
  print RepSpacing, numpy.array([i, j, k])
  cornerpos = numpy.array([i, j, k]) * RepSpacing

  delta1 = realize(A.power, A.seedtable[i, j, k], A.NmeshEff, 
               A.Nmesh, A.BoxSize, A.Nmesh)
  losdisp1 = realize_losdisp(A.power, cornerpos,
               A.seedtable[i, j, k], A.NmeshEff,
               A.Nmesh, A.BoxSize, A.Nmesh)
  coarse = pixels['pos'] / A.BoxSize * A.Nmesh
  fine = (pixels['pos'] / A.BoxSize * A.Nrep - [i, j, k]) * A.Nmesh
  d0 = map_coordinates(delta0, coarse.T, mode='wrap')
  disp0 = map_coordinates(losdisp0, coarse.T, mode='wrap', order=5)
  d1 = map_coordinates(delta1, fine.T, mode='wrap')
  disp1 = map_coordinates(losdisp1, fine.T, mode='wrap', order=5)
  pixels['delta'] = d1 + d0
  pixels['losdisp'] = disp0 + disp1
  return pixels

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
  #print Nmesh, BoxSize, Nsample, K0, 'kmax = ', Nsample / 2 * K0
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
      K0Coarse = 2 * numpy.pi / BoxSize
      Kthresh = K0Coarse * NmeshCoarse / 2
      thresh = Kthresh / K0
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
      disp[i][numpy.isnan(disp[i])] = 0
    losdisp += disp
  return losdisp

main()

