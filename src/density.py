import numpy

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
      [ - 0.5 * NmeshCoarse * K0, 0.5 * NmeshCoarse * K0]
      in the middle of the power spectrum,
      presumbaly power from region will be added as large
      scale power from a different realization, with interpolation.
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
      thresh = NmeshCoarse * Nsample / Nmesh / 2 * 0.7
      if i < thresh and i > - thresh:
        mask = (j < thresh) & (k < thresh) & \
               (j > - thresh) & (k > -thresh)
        gauss[i][mask] = 0
  
  gauss[Nsample / 2, ...] = 0
  gauss[:, Nsample / 2, :] = 0
  gauss[:, :, Nsample / 2] = 0

  delta = numpy.fft.irfftn(gauss)
  # fix the fftpack normalization
  delta *= Nsample ** 3

  return numpy.float32(delta)

def lognormal(delta, std=None, out=None):
    # lognormal transformation
    if out is None:
      out = delta.copy()
    else:
      out[...] = delta
    if std is None:
      std = (out ** 2).mean() ** 0.5
    out -= std * std * 0.5
    numpy.exp(out, out=out)
    out -= 1  
    return out

def realize_dispr(PowerSpec, cornerpos,
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
      r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
      disp[i][r!=0] /= r[r!=0]
      disp[i][r==0] = 0
    losdisp += disp
  return losdisp

def realize_dispz(PowerSpec, cornerpos,
            seed, Nmesh, Nsample, BoxSize, 
            NmeshCoarse=None):
  dispkernel = [
    lambda kx, ky, kz, k: 1j * kx * k ** -2,
    lambda kx, ky, kz, k: 1j * ky * k ** -2,
    lambda kx, ky, kz, k: 1j * kz * k ** -2]
  ax = 2
  losdisp = realize(PowerSpec, seed=seed, Nmesh=Nmesh, 
         Nsample=Nsample, BoxSize=BoxSize, NmeshCoarse=NmeshCoarse,
         kernel=dispkernel[ax])
  return -losdisp
