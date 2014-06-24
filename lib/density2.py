import numpy
from scipy.interpolate import interp1d
import pyfftw

from lazy import Lazy
class InPlaceRealFFT(object):
    def __init__(self, shape, dtype):
        """ initialize a inplace real FFT with type dtype """
        dtype = numpy.dtype(dtype)
        if dtype == numpy.dtype('f8'):
            compdtype = numpy.dtype('complex128')
        elif dtype == numpy.dtype('f4'):
            compdtype = numpy.dtype('complex64')
        else:
            raise TypeError("dtype has to be f4 or f8")
        compshape = list(shape)
        compshape[-1] = compshape[-1] // 2 + 1
        buffer = pyfftw.n_byte_align_empty(compshape, 4096, compdtype, order='C')

        self.real = buffer.view(dtype=dtype)[..., :shape[-1]]
        self.complex = buffer

        print self.real.shape
        print self.complex.shape
        print 'planning fftw'
        self.r2cPlan = pyfftw.FFTW(self.real, self.complex, axes=range(len(self.real.shape)),
                direction='FFTW_FORWARD', flag=['FFTW_ESTIMATE'])
        self.c2rPlan = pyfftw.FFTW(self.complex, self.real, axes=range(len(self.real.shape)),
                direction='FFTW_BACKWARD', flag=['FFTW_ESTIMATE'])
        print 'done planning fftw'

    def r2c(self):
        self.r2cPlan.execute()

    def c2r(self):
        self.c2rPlan.execute()

    @Lazy
    def complexorder(self):
        k = numpy.fft.fftfreq(self.real.shape[0])
        k2 = 0
        Nd = len(self.real.shape)
        for d in range(len(self.real.shape)):
            shape = numpy.repeat(1, Nd)
            shape[d] = self.complex.shape[d]
            k2 = numpy.maximum(k2, numpy.abs(k[:shape[d]].reshape(shape)))
        return k2.ravel().argsort(kind='merge') # stable

    def fillcomplex(self, func):
        """ fill the complex array from numbers obtained from func
            def func(N):
                return N complex numbers
            The purpose of this function is to ensure an inside out filling from
            small K to large K, so that as we increase the resolution the large
            scale power is conserved.
        """
        arg = self.complexorder
        chunksize = 1024 * 1024
        for i in range(0, len(arg), chunksize):
            sl = slice(i, i + chunksize)
            ind = arg[sl]
            self.complex.put(ind, func(len(ind)))

class Density(object):
    def __init__(self, Nsample, power, BoxSize, Kmin=None, Kmax=None):
        self.fft = InPlaceRealFFT((Nsample, ) * 3, 'f4')
        self.PowerSpec = interp1d(power[0], power[1], kind='linear', copy=True, 
                bounds_error=False, fill_value=0)
        self.BoxSize = BoxSize
        self.Kmin = Kmin
        self.Kmax = Kmax

    def fill(self, seed, kernel):
        rng = numpy.random.RandomState(seed)
        def g(N):
            return rng.normal(scale=2 ** -0.5, size=N* 2
                ).view(dtype=numpy.complex128)

        self.fft.fillcomplex(g)

        Nsample = self.fft.real.shape[0]
        
        K0 = 2 * numpy.pi / self.BoxSize
        Ktab = K0 * numpy.fft.fftfreq(Nsample, 1. / Nsample)

        deltak = self.fft.complex

        for i in range(Nsample):
            # loop over i to save memory
            j, k = numpy.ogrid[0:Nsample, 0:Nsample/2+1]
            K = numpy.empty((Nsample, Nsample/2 + 1), dtype='f4')
            Kx, Ky, Kz = Ktab[i], Ktab[j], Ktab[k]
            K[...] = (Kx ** 2 + Ky ** 2 + Kz ** 2) ** 0.5
            PK = self.PowerSpec(K) ** 0.5
            if self.Kmin:
#                lowcut = (Kx > self.Kmin) & (Ky > self.Kmin) & (Kz > self.Kmin)
                # digging the hole
                lowcut = K >= self.Kmin
              #1 - numpy.exp( - (K / Kmin) ** 2)
             #   thresh = Kmin / K0
             #   if i < thresh and i > - thresh:
             #     mask = (j < thresh) & (k < thresh) & \
                     #            (j > - thresh) & (k > -thresh)
             #     deltak[i][mask] = 0
                deltak[i] *= lowcut
            if self.Kmax:
                #highcut = (Kx <= self.Kmax) & (Ky <= self.Kmax) & (Kz <= self.Kmax)
                highcut = K <= self.Kmax
                #numpy.exp( - (K / Kmax) ** 2)
                deltak[i] *= highcut
            deltak[i] *= (PK * K0 ** 1.5)
            if kernel is not None:
                deltak[i] *= kernel(Kx, Ky, Kz, K)
            deltak[i][numpy.isnan(deltak[i])] = 0
      
        # more FFT conjugate requirements
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
      
        # to be safe
        deltak[h, ...] = 0
        deltak[:, h, :] = 0
        deltak[..., h] = 0

    def realize(self):
        self.fft.c2r()
        return self.fft.real

    def power(self):
        self.fft.r2c()
        return self.fft.complex

def test():
    power = numpy.loadtxt('power.txt', unpack=True)
    density = Density(Nsample=128, seed=128, power=power, BoxSize=10000., Kmin=None, Kmax=None)
    density.fill(kernel=None)
    c1 = density.fft.complex.copy()
    d1 = density.realize().copy()
    p1 = density.power().copy()

    density = Density(Nsample=64,seed=128, power=power, BoxSize=10000., Kmin=None, Kmax=None)

    density.fill(kernel=None)
    c2 = density.fft.complex.copy()
    d2 = density.realize().copy()
    p2 = density.power().copy()

    return d1, c1, d2, c2
