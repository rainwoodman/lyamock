import numpy
from splat import splat
import sharedmem
from args import pixeldtype, bitmapdtype
from cosmology import interp1d
from scipy.optimize import leastsq

def deltatotau(A, delta, tau, Zreal):
    afactor = findafactor(A, delta, Zreal)
    chunksize = 1024 * 1024 * 4
    def work(i):
        s = slice(i, i + chunksize)
        tau[s] =-afactor(Zreal[s])* numpy.exp(delta[s])
    with sharedmem.Pool() as pool:
        pool.map(work, range(0, len(Zreal), chunksize))

def findafactor(A, delta, Zreal):
    Nbins = 15

    Rmin = A.sightlines.Rmin.min()
    Rmax = A.sightlines.Rmax.max()
    Rbins = numpy.linspace(Rmin, Rmax, Nbins + 1, endpoint=True)
    zbins = 1 / A.cosmology.Dc.inv(Rbins / A.DH) - 1
    afactor = numpy.empty(Nbins)
    Rcenter = (Rbins[1:] + Rbins[:-1]) * 0.5
    Zcenter = 1 / A.cosmology.Dc.inv(Rcenter / A.DH) - 1
    afactor[-1] = 1.0
    Zreal = Zreal[numpy.random.uniform(size=len(Zreal)) < 0.01]
    dig = numpy.digitize(Zreal, zbins)
    for i in range(len(Rbins) - 1):
        d = delta[dig == i + 1]
        print Zcenter[i], 'npixels=', len(d)
        a = A.cosmology.Dc.inv(Rcenter[i] / A.DH)
        expected = A.FPGAmeanflux(a)
        afactor[i] = findoneafactor(A, d, expected, afactor[i-1])
        print Zcenter[i], 'afactor', afactor[i]
    AZ = interp1d(Rcenter, afactor, fill_value=1.0, kind=4)
    numpy.savez('afactor.npz', R=Rcenter, a=afactor)
    return AZ

def findoneafactor(A, delta, expected, hint):
    if len(delta) == 0:
        return 1.0
    def cost(afactor):
        chunksize = 1024 * 16
        def work(i):
            s = slice(i, i + chunksize)
            rawflux = numpy.exp(-afactor * numpy.exp((1 + delta[s])))
            rawflux[numpy.isnan(rawflux)] = 0
            return rawflux.sum(dtype='f8')
        with sharedmem.MapReduce() as pool:
            total = numpy.sum(pool.map(work, range(0, len(delta), chunksize)))
        return total / len(delta) - expected
    p = leastsq(cost, hint, full_output=False)
    return p[0]

def main(A):
    """matched mean flux for dreal"""

    delta = A.P('delta')
    Zreal = A.P('Zreal')
    tau = A.P('tau', memmap='w+', shape=delta.shape)
    deltatotau(A, delta, tau, Zreal)
    tau.flush()
