import numpy
import sharedmem
from scipy.optimize import leastsq
from args import pixeldtype
from cosmology import interp1d

def main(A):
    """find the normalization factor matching the mean flux"""
  
    if A.RedshiftDistortion:
        Z = A.P('Zred', memmap='r')
    else:
        Z = A.P('Zreal', memmap='r')
  
    rawflux = A.P('rawflux', memmap='r')
    zbins = numpy.linspace(Z.min(), Z.max(), 100, endpoint=True)
    zcenter = 0.5 * (zbins[1:] + zbins[:-1])
    afactor = numpy.ones_like(zcenter)
    meanflux_expected = A.FPGAmeanflux(1 / (1 + zcenter))
  
    print 'zbins = ', zbins[0], zbins[-1]

    dig = numpy.digitize(Z, zbins)
    ind = dig.argsort()
    dig = dig[ind]
  
    with sharedmem.Pool(use_threads=True) as pool:
        for i in range(len(zcenter)):
            left = dig.searchsorted(i + 1, side='left')
            right = dig.searchsorted(i + 1, side='right')
            subset = ind[left:right]
            if subset.size == 0:
                afactor[i] = 1.0
                continue

            subset.sort()
            F = rawflux[subset]
            def cost(az):
                def work(F):
                    return (F[F!=0] ** az[0]).sum(dtype='f8')
                Fsum = numpy.sum(pool.starmap(work, pool.zipsplit((F,))))
                Fmean = Fsum / F.size
                dist = (Fmean - meanflux_expected[i])
                return dist
            if i > 1:
              p0 = afactor[i - 1]
            else:
              p0 = 1.0
            p = leastsq(cost, p0, full_output=False)
            afactor[i] = p[0]
            print i, zcenter[i], afactor[i], meanflux_expected[i], F.size
  
    numpy.savetxt(A.datadir + '/afactor.txt', zip(zcenter, afactor),
            fmt='%g')

    print 'normalizing'
    AZ = interp1d(zcenter, afactor, fill_value=1.0, kind=4)
    chunksize = 1048576
    fluxfile = A.P('flux', justfile='w')
    for i in range(0, len(rawflux), chunksize):
        SL = slice(i, i + chunksize)
        flux = numpy.zeros(rawflux[SL].shape, dtype=pixeldtype['flux'])
        mask = rawflux[SL] != 0
        flux[mask] = rawflux[SL][mask] ** AZ(Z[SL])[mask]
        flux.tofile(fluxfile)
        fluxfile.flush()
