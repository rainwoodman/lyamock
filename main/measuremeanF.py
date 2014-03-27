import numpy
import sharedmem

from common import Config
from common import MeanFractionModel
from common import VarFractionModel
from common import Sightlines
from common import SpectraOutput

from lib.lazy import Lazy
from lib.chunkmap import chunkmap


def main(A):
    sightlines = Sightlines(A)
    spectra = SpectraOutput(A)
    loglam = sharedmem.empty(sightlines.Npixels.sum(), 'f4')
    chunksize = 100

    #sharedmem.set_debug(True)
    def work(i):
        sl = slice(sightlines.PixelOffset[i],
            sightlines.PixelOffset[i] + sightlines.Npixels[i])
        loglam[sl] = spectra.LogLam[i]
    chunkmap(work, range(len(sightlines)), chunksize=100)

    Nbins = 8
    zBins = numpy.linspace(2.0, 4.0, Nbins + 1, endpoint=True)
    LogLamBins = numpy.log10(1216.0 * (1 + zBins ))
    z = 0.5 * (zBins[1:] + zBins[:-1])

    ind = numpy.digitize(loglam, LogLamBins)
    N = numpy.bincount(ind, minlength=Nbins+2)
    N[N == 0] = 1.0
    F = numpy.exp(-spectra.taured.data)
    K1 = numpy.bincount(ind, F, minlength=Nbins+2) / N
    K2 = numpy.bincount(ind, F ** 2, minlength=Nbins+2) / N
    meanF = K1
    varF = K2 - K1 ** 2
    meanF = meanF[1:-1]
    varF = varF[1:-1]

    F = numpy.exp(-spectra.taureal.data)
    K1 = numpy.bincount(ind, F, minlength=Nbins+2) / N
    K2 = numpy.bincount(ind, F ** 2, minlength=Nbins+2) / N
    meanFreal = K1
    varFreal = K2 - K1 ** 2
    meanFreal = meanFreal[1:-1]
    varFreal = varFreal[1:-1]
    print z
    print meanF
    print varF
    numpy.savez(A.MeasureMeanFractionOutput, 
            a=1/(1+z), xmeanF=meanF, xvarF=varF, 
            xmeanFreal=meanFreal,
            xvarFreal=varFreal)

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
