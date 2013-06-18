import numpy
from scipy.ndimage import spline_filter
from scipy.integrate import romberg
from snakefill import snakefill

def realize(PowerSpec,
            seed, Nsample, BoxSize, 
            Kmin=None,
            Kmax=None,
            kernel=lambda kx, ky, kz, k: 1,
            order=False):
  """
      realize a powerspectrum.
      PowerSpec must be in gadget unit, (2 pi kpc/h) ** 3

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
  rng = numpy.random.RandomState(seed)
  K0 = 2 * numpy.pi / BoxSize
  A = snakefill((Nsample, Nsample, Nsample / 2 + 1))
  A = A.ravel().argsort()
  gauss = rng.normal(scale=2 ** -0.5, 
            size=(Nsample, Nsample, Nsample / 2 + 1, 2)
          ).view(dtype=numpy.complex128).reshape(Nsample, Nsample, Nsample / 2 + 1)
  deltak = numpy.empty_like(gauss)
  deltak.ravel()[A] = gauss.ravel()
  del gauss, A
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
    deltak[i] *= (PK * K0 ** 1.5) * kernel(i * K0, j * K0, k * K0, K)
    deltak[i][numpy.isnan(deltak[i])] = 0
  
  deltak[Nsample / 2, ...] = 0
  deltak[:, Nsample / 2, :] = 0
  deltak[:, :, Nsample / 2] = 0

  delta = numpy.fft.irfftn(deltak)
  del deltak
  # fix the fftpack normalization
  delta *= Nsample ** 3
  var = delta.var()
  if order is not False:
    delta = spline_filter(delta, order=order)
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

def realize_dispr(PowerSpec, cornerpos, observerpos,
        seed, Nsample, BoxSize, 
        Kmin=None, Kmax=None, order=False):
    dispkernel = [
            lambda kx, ky, kz, k: 1j * kx * k ** -2,
            lambda kx, ky, kz, k: 1j * ky * k ** -2,
            lambda kx, ky, kz, k: 1j * kz * k ** -2]
    losdisp = numpy.zeros((Nsample, Nsample, Nsample), dtype='f4')
    for ax in range(3):
        disp, var = realize(PowerSpec, seed=seed,  
              Nsample=Nsample, BoxSize=BoxSize, 
              Kmin=Kmin, Kmax=Kmax,
              kernel=dispkernel[ax])
        for i in range(Nsample):
            j, k = numpy.ogrid[0:Nsample, 0:Nsample]
            x = cornerpos[0] + i * BoxSize - observerpos[0]
            y = cornerpos[1] + j * BoxSize - observerpos[1]
            z = cornerpos[2] + k * BoxSize - observerpos[2]
            disp[i] *= ([-x, -y, -z][ax])
            r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
            disp[i][r!=0] /= r[r!=0]
            disp[i][r==0] = 0
        losdisp += disp
        del disp

    if order is not False:
        losdisp = spline_filter(losdisp, order=order)
    return losdisp

def realize_dispz(PowerSpec, cornerpos, observerpos,
            seed, Nsample, BoxSize, 
            Kmin=None, Kmax=None, order=False):
    """ observerpos is unused."""
    dispkernel = [
      lambda kx, ky, kz, k: 1j * kx * k ** -2,
      lambda kx, ky, kz, k: 1j * ky * k ** -2,
      lambda kx, ky, kz, k: 1j * kz * k ** -2]
    ax = 2
    losdisp, var = realize(PowerSpec, seed=seed, 
           Nsample=Nsample, 
           BoxSize=BoxSize, Kmin=Kmin, Kmax=Kmax,
           kernel=dispkernel[ax])
    losdisp *= -1
    if order is not False:
        losdisp = spline_filter(losdisp, order=order)
    return losdisp

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


