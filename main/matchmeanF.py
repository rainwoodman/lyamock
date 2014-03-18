import numpy
import sharedmem
from scipy.optimize import brentq
from scipy.integrate import romberg

from args import Config
from args import MeanFractionModel
from args import VarFractionModel
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

    Nbins = 8
    zBins = numpy.linspace(2.0, 4.0, Nbins + 1, endpoint=True)
    
    z = 0.5 * (zBins[1:] + zBins[:-1])
    A = numpy.empty_like(z)
    for i in range(Nbins):
        A[i], E, V = fitRange(zBins[i], zBins[i + 1], spectra.taured)
        print A[i], z[i], E, meanfractionmodel(1 / (1 + z[i])), V, varfractionmodel(1 / (1 + z[i]))
    print A, z

def fitRange(zMin, zMax, field):
    # field needs to be spectra.taured or spectra.taureal

    # now lets pick the pixels
    LogLamMin = numpy.log10((zMin + 1) * 1216.)
    LogLamMax = numpy.log10((zMax + 1) * 1216.)

    Npixels = numpy.empty(len(spectra), 'intp')

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

    # What is the mean of the model?
    aMin = 1 / (1. + zMin)
    aMax = 1 / (1. + zMax)

    data = romberg(meanfractionmodel, aMin, aMax) / (aMax - aMin)

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
