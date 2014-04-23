import numpy
import sharedmem

from common import Config
from common import PowerSpectrum

from sys import argv
from lib import density
from lib import density2

def main():
    A = Config(argv[1])
    var = initlya(A)

    D2 = A.cosmology.Dplus(1 / 3.0) / A.cosmology.Dplus(1.0)
    D3 = A.cosmology.Dplus(1 / 4.0) / A.cosmology.Dplus(1.0)

    print 'lya field', 'var', var
    print 'growth factor at z=2.0, 3.0', D2, D3
    print 'lya variance adjusted to z=2.0, z=3.0', D2 ** 2 * var, D3 **2 * var

def initlya(A):
    print 'measuring Lya cloud scale fluctuation with NBox Boxes'
    powerspec = PowerSpectrum(A)
    NBox = 8
    # fill the lya resolution small boxes

    cutoff = 0.5 * 2 * numpy.pi / A.BoxSize * A.NmeshEff

    den3 = density2.Density(A.NmeshLyaBox, power=powerspec,
            BoxSize=A.BoxSize/A.NmeshEff,
            Kmin=cutoff)

    def kernel(kx, ky, kz, k):
        f2 = 1 / (1 + (A.LogNormalScale * k) ** 2)
        return f2

    if A.NmeshLyaBox > 1:
        # this is the kernel to correct for log normal transformation
        # as it matters when we get to near the JeansScale
  
        seed = A.RNG.randint(1<<21 - 1, size=NBox)
        def work(i):
            den3.fill(seed=seed[i], kernel=kernel)
            deltalya = den3.realize()
            return deltalya.var(dtype='f8')

        with sharedmem.Pool() as pool:
            var = numpy.mean(pool.map(work, range(NBox)))

    return var

if __name__ == '__main__':
    main()
