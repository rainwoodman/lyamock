import numpy
import sharedmem
from args import Config
from args import PowerSpectrum

from sys import argv
from lib import density
def main():
    A = Config(argv[1])
    print initlya(A)

def initlya(A):
    print 'measuring Lya cloud scale fluctuation with NBox Boxes'
    powerspec = PowerSpectrum(A)
    NBox = 8
    # fill the lya resolution small boxes
    deltalya = sharedmem.empty((NBox, 
        A.NmeshLyaBox, A.NmeshLyaBox, A.NmeshLyaBox))
  
    deltalya[:] = 0
    if A.NmeshLyaBox > 1:
        # this is the kernel to correct for log normal transformation
        # as it matters when we get to near the JeansScale
        def kernel(kx, ky, kz, k):
            f2 = 1 / (1 + (A.LogNormalScale / (2 * numpy.pi) * k) ** 2)
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

    print 'lya field', 'mean', deltalya.mean(), 'var', deltalya.var()
    return deltalya.var()

if __name__ == '__main__':
    main()
