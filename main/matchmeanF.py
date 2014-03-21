import numpy
import sharedmem
from scipy.optimize import brentq
from scipy.integrate import romberg

from common import Config
from common import MeanFractionModel
from common import VarFractionModel
from sightlines import Sightlines
from convolve import SpectraOutput

def main(A):
    """ match the mean fraction by fixing prefactor A(a) on tau
        add in thermal broadening and redshift distortion """
    global spectra
    global meanfractionmodel
    global varfractionmodel
    spectra = SpectraOutput(A)
    varfractionmodel = VarFractionModel(A)
    meanfractionmodel = MeanFractionModel(A)
    fname = A.datadir + '/matchmeanFoutput.npz'

    Nbins = 64
    zBins = numpy.linspace(2.0, 4.0, Nbins + 1, endpoint=True)
    LogLamBins = numpy.log10(1216.0 * (1 + zBins ))
    z = 0.5 * (zBins[1:] + zBins[:-1])
    Af = sharedmem.empty(z.shape)
    V = sharedmem.empty(z.shape)
    E = sharedmem.empty(z.shape)
    with sharedmem.MapReduce() as pool:
        def work(i):
            Af[i], E[i], V[i] = fitRange(LogLamBins[i], LogLamBins[i + 1], spectra.taured)
            with pool.ordered:
                print Af[i], z[i], E[i], meanfractionmodel(1 / (1 + z[i])), V[i], varfractionmodel(1 / (1 + z[i]))
        pool.map(work, range(Nbins))
    numpy.savez(fname, Af=Af, z=z, V=V, E=E)
    print Af, z

def fitRange(LogLamMin, LogLamMax, field):
    # field needs to be spectra.taured or spectra.taureal
    Npixels = numpy.empty(len(spectra), 'intp')

    # What is the mean of the model?
    def fun(loglam):
        a = 1216. / 10 ** loglam 
        return meanfractionmodel(a)
    data = romberg(fun, LogLamMin, LogLamMax) / (LogLamMax - LogLamMin)
    print LogLamMin, LogLamMax, data, 

    for i in range(len(spectra)):
        LogLam = spectra.LogLam[i]
        mask = (LogLam >= LogLamMin) & (LogLam <= LogLamMax)
        Npixels[i] = mask.sum()

    values = numpy.empty(Npixels.sum(), 'f8')
    PixelOffset = numpy.concatenate([[0], Npixels.cumsum()[:-1]])

    for i in range(len(spectra)):
        LogLam = spectra.LogLam[i]
        mask = (LogLam >= LogLamMin) & (LogLam <= LogLamMax)
        sl = slice(PixelOffset[i], PixelOffset[i] + Npixels[i])
        values[sl] = field[i][mask]

    # OK we have collected the tau values now use brentq to 
    # solve for A.
    def func(afac):
        return numpy.exp(-afac * values).mean() / data - 1.0
    
    afac = brentq(func, 0, 1e-2)
    fraction = numpy.exp(-afac * values)

    return afac, fraction.mean(), fraction.var()

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
