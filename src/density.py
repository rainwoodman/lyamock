import numpy
from scipy.ndimage import spline_filter
from scipy.integrate import romberg
from snakefill import snakefill
from scipy.interpolate import interp1d

def realize(power,
            seed, Nsample, 
            BoxSize, 
            Kmin=None,
            Kmax=None,
            disp=False,
            kernel=None,
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
  rng = numpy.random.RandomState(seed)
  K0 = 2 * numpy.pi / BoxSize
  gauss = rng.normal(scale=2 ** -0.5, 
            size=(Nsample, Nsample, Nsample / 2 + 1, 2)
          ).view(dtype=numpy.complex128).reshape(Nsample, Nsample, Nsample / 2 + 1)

  shuffle = snakefill((Nsample, Nsample, Nsample / 2 + 1))
  shuffle = shuffle.ravel().argsort()
  deltak = numpy.empty_like(gauss)
  deltak.ravel()[shuffle] = gauss.ravel()
  del gauss, shuffle

  #deltak = gauss
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
      lowcut = K >= Kmin
      #1 - numpy.exp( - (K / Kmin) ** 2)
   #   thresh = Kmin / K0
   #   if i < thresh and i > - thresh:
   #     mask = (j < thresh) & (k < thresh) & \
   #            (j > - thresh) & (k > -thresh)
   #     deltak[i][mask] = 0
      deltak[i] *= lowcut
    if Kmax:
      highcut = K <= Kmax
      #numpy.exp( - (K / Kmax) ** 2)
      deltak[i] *= highcut
    deltak[i] *= (PK * K0 ** 1.5)
    if kernel is not None:
      deltak[i] *= kernel(i * K0, j * K0, k * K0, K)
    if disp is not False:
      deltak[i] *= dispkernel[disp](i * K0, j * K0, k * K0, K)
    deltak[i][numpy.isnan(deltak[i])] = 0
  
  deltak[Nsample / 2, ...] = 0
  deltak[:, Nsample / 2, :] = 0
  deltak[:, :, Nsample / 2] = 0

  delta = numpy.fft.irfftn(deltak)
  del deltak
  # fix the fftpack normalization
  delta *= Nsample ** 3
  var = delta.var()
  return numpy.float32(delta), var

def lognormal(delta, std, out=None):
    """mean(delta)  shall be zero"""
    # lognormal transformation
    if out is None:
      out = delta.copy()
    else:
      out[...] = delta
    out -= std * std * 0.5
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


