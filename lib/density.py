import numpy
from scipy.integrate import romberg
from snakefill import snakefill
from scipy.interpolate import interp1d

import pyfftw

class begin_irfftn(numpy.ndarray):
    def __new__(cls, shape, dtype):
        """ shape is the shape of the real output,
            call finish_irfftn to get the output
        """
        newshape = list(shape)
        newshape[-1] = shape[-1] // 2 + 1
        input = pyfftw.n_byte_align_empty(newshape, 64, dtype, order='C')
        input = input.view(type=cls)
        if shape[-1] % 2 == 0:
            output = input.view(dtype=input.real.dtype)[..., :-2]
        else:
            output = input.view(dtype=input.real.dtype)[..., :-1]
        input.output = output
        input.plan = pyfftw.FFTW(input, output,
                axes=range(len(shape)),
                direction='FFTW_BACKWARD')
        return input

def build_shuffle(shape):
    shuffle = snakefill(shape)
    shuffle = shuffle.ravel().argsort()
    if numpy.product(shape) < 1024 ** 3:
        shuffle = numpy.int32(shuffle)
    return shuffle

def finish_irfftn(input):
    input.plan.execute()
    output = input.output
    del input.plan
    del input.output
    return output

def execute_irfftn(input):
    input.plan.execute()
    output = input.output
    return output

def gaussian(deltak, shuffle, seed):
    rng = numpy.random.RandomState(seed)
    chunksize = 1024 * 1024 
    for i in range(0, len(shuffle), chunksize):
        s = slice(i, i + chunksize)
        ss = shuffle[s]
        gausstmp = rng.normal(scale=2 ** -0.5, size=len(ss)* 2
            ).view(dtype=numpy.complex128)
        deltak.put(ss, gausstmp)
gengaussian = gaussian

def realize(power,
            seed, Nsample, 
            BoxSize, 
            Kmin=None,
            Kmax=None,
            disp=False,
            kernel=None,
            gaussian=None,
            return_deltak=False
            ):
  """
      realize a powerspectrum.
      power (k, p) must be in gadget unit, (2 pi kpc/h) ** 3
    
      default is to make delta field.

      to get displacement field, use
      lambda kx, ky, kz, k: I * kx * k ** -2

      if Kmin is not None, leave out a region of
      [ - Kmin, Kmin], 
      in the middle of the power spectrum,

      In integer, this is Kmin / K0, where
      K0 = 2 pi / BoxSize

      presumbaly power from region will be added as large
      scale power from a different realization, with interpolation.
  """
  dispkernel = [
            lambda kx, ky, kz, k: 1j * kx * k ** -2,
            lambda kx, ky, kz, k: 1j * ky * k ** -2,
            lambda kx, ky, kz, k: 1j * kz * k ** -2
  ]
  PowerSpec = interp1d(power[0], power[1], kind='linear', copy=True, 
                         bounds_error=False, fill_value=0)
  K0 = 2 * numpy.pi / BoxSize
  if gaussian is None:
      gaussian = begin_irfftn((Nsample, Nsample, Nsample),
            dtype=numpy.complex64)
      shuffle = build_shuffle((Nsample, Nsample, Nsample //2 + 1))
      gengaussian(gaussian, shuffle, seed)
      del shuffle

  deltak = gaussian
  # convolve gauss with powerspec to delta_k, stored in deltak.
  for i in range(Nsample):
    # loop over i to save memory
    if i > Nsample / 2: i = i - Nsample
    j, k = numpy.ogrid[0:Nsample, 0:Nsample/2+1]
    j[j > Nsample / 2] = j[j > Nsample / 2] - Nsample
    K = numpy.empty((Nsample, Nsample/2 + 1), dtype='f4')
    K[...] = (i ** 2 + j ** 2 + k ** 2) ** 0.5 * K0
    PK = PowerSpec(K) ** 0.5
    if Kmin:
      # digging the hole
 #     lowcut = K >= Kmin (spherical)
      lowcut = (i * K0 >= Kmin) & (j * K0 >= Kmin) & (k * K0 >= Kmin)
      #1 - numpy.exp( - (K / Kmin) ** 2)
   #   thresh = Kmin / K0
   #   if i < thresh and i > - thresh:
   #     mask = (j < thresh) & (k < thresh) & \
   #            (j > - thresh) & (k > -thresh)
   #     deltak[i][mask] = 0
      deltak[i] *= lowcut
    if Kmax:
      #highcut = K <= Kmax
      highcut = (i * K0 <= Kmax) & (j * K0 <= Kmax) & (k * K0 <= Kmax)
      #numpy.exp( - (K / Kmax) ** 2)
      deltak[i] *= highcut
    deltak[i] *= (PK * K0 ** 1.5)
    if kernel is not None:
      deltak[i] *= kernel(i * K0, j * K0, k * K0, K)
    if disp is not False:
      deltak[i] *= dispkernel[disp](i * K0, j * K0, k * K0, K)
    deltak[i][numpy.isnan(deltak[i])] = 0
  
  h = Nsample // 2
  j = numpy.arange(1, h)
  for i in range(1, h):
      deltak[Nsample - i, Nsample - j, h] = numpy.conjugate(deltak[i, j, h])
      deltak[Nsample - i, Nsample - j, 0] = numpy.conjugate(deltak[i, j, 0])
      deltak[Nsample - i, j, 0] = numpy.conjugate(deltak[i, Nsample - j, 0])
      deltak[i, Nsample - j, 0] = numpy.conjugate(deltak[Nsample - i, j, 0])
      deltak[Nsample - i, 0, 0] = numpy.conjugate(deltak[i, 0, 0])
      deltak[0, Nsample - i, 0] = numpy.conjugate(deltak[0, i, 0])

  deltak.imag[0, 0, 0] = 0

  deltak[h, ...] = 0
  deltak[:, h, :] = 0
  deltak[..., h] = 0

  if return_deltak: 
      deltak_copy = deltak.copy()
  delta = execute_irfftn(deltak)
  # fix the fftpack normalization
  # no need for fftw delta *= Nsample ** 3
  if return_deltak:
      return delta, deltak_copy
  var = delta.var(dtype='f8')
  return delta, var

def lognormal(delta, std, out=None):
    """mean(delta)  shall be zero"""
    # lognormal transformation
    if out is None:
      out = delta.copy()
    else:
      out[...] = delta
    #sigma = numpy.log(std ** 2 + 1) ** 0.5
    #mu =  - sigma ** 2 / 2
    #out *= sigma / std
    mu = - std ** 2 * 0.5
    out += mu
    numpy.exp(out, out=out)
    out -= 1  
    return out

def sigma(power, R):
    def integrand(k):
        kr = R * k
        kr2 = kr ** 2
        kr3 = kr ** 3

        w = 3 * (numpy.sin(kr) / kr3 - numpy.cos(kr) / kr2)
        x = 4 * numpy.pi * k * k * w * w * power(k)
        if numpy.isscalar(x):
            if numpy.isnan(x): return 0
        else:
            x[numpy.isnan(x)] = 0
        return x
    return romberg(integrand, 0, 1000. / R, tol=0, rtol=1e-9, divmax=100,
            vec_func=True) ** 0.5


