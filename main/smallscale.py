import numpy
import sharedmem

from common import Config
from common import PowerSpectrum

from sys import argv
from lib import density

def main():
    A = Config(argv[1])
    print initlya(A)

def initlya(A):
    print 'measuring Lya cloud scale fluctuation with NBox Boxes'
    powerspec = PowerSpectrum(A)
    NBox = 1
    # fill the lya resolution small boxes
    deltalya = sharedmem.empty((NBox, 
        A.NmeshLyaBox, A.NmeshLyaBox, A.NmeshLyaBox))
  
    deltalya[:] = 0
    if A.NmeshLyaBox > 1:
        # this is the kernel to correct for log normal transformation
        # as it matters when we get to near the JeansScale
        def kernel(kx, ky, kz, k):
            f2 = 1 / (1 + (A.LogNormalScale * k) ** 2)
            return f2
  
        cutoff = 0.5 * 2 * numpy.pi / A.BoxSize * A.NmeshEff
        seed = A.RNG.randint(1<<21 - 1, size=NBox)
        def work(i):
            deltalya[i], varjunk = density.realize(powerspec, 
                    seed[i],
                    A.NmeshLyaBox, 
                    A.BoxSize / A.NmeshEff, 
                    Kmin=cutoff,
                    kernel=kernel)
        with sharedmem.Pool() as pool:
            pool.map(work, range(NBox))

    D2 = A.cosmology.Dplus(1 / 3.0) / A.cosmology.Dplus(1.0)
    D3 = A.cosmology.Dplus(1 / 4.0) / A.cosmology.Dplus(1.0)
    print 'lya field', 'mean', deltalya.mean(), 'var', deltalya.var()
    print 'growth factor at z=2.0, 3.0', D2, D3
    Var = deltalya.var()
    print 'lya variance adjusted to z=2.0, z=3.0', D2 ** 2 * Var, D3 **2 * Var
    return Var

if __name__ == '__main__':
    main()
