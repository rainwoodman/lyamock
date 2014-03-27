import numpy
import sharedmem
from scipy.optimize import brentq, root
from scipy.integrate import romberg

from common import Config
from common import MeanFractionModel
from common import VarFractionModel
from common import Sightlines

from lib.lazy import Lazy
from lib.chunkmap import chunkmap

from spectra import SpectraMaker

def main(A):
    """ match the mean fraction by fixing prefactor A(a) and B(a) on tau
        requires 'gaussian' to be finished.
        run before convolve, though it uses functions in convolve for evaluating 
        the cost function.
    """
    global meanfractionmodel
    global varfractionmodel

    varfractionmodel = VarFractionModel(A)
    meanfractionmodel = MeanFractionModel(A)

    Nbins = 8
    zBins = numpy.linspace(2.0, 4.0, Nbins + 1, endpoint=True)
    LogLamBins = numpy.log10(1216.0 * (1 + zBins ))
    z = 0.5 * (zBins[1:] + zBins[:-1])
    Af = sharedmem.empty(z.shape)
    Bf = sharedmem.empty(z.shape)
    xmeanF = sharedmem.empty(z.shape)
    xstdF = sharedmem.empty(z.shape)
    def work(i):
        if i > 0:
            Afguess, Bfguess = Af[i-1], Bf[i-1]
        else:
            Afguess, Bfguess = (0.00015, 1.5)
        Af[i], Bf[i], xmeanF[i], xstdF[i] = fitRange(A, LogLamBins[i], LogLamBins[i + 1], 
                Afguess, Bfguess)
    map(work, range(Nbins))
    numpy.savez(A.MatchMeanFractionOutput, a=1 / (z+1), 
            Af=Af, Bf=Bf, xmeanF=xmeanF, xvarF=xstdF ** 2)

def fitRange(A, LogLamMin, LogLamMax, Afguess, Bfguess):
    sightlines = Sightlines(A, LogLamMin, LogLamMax)
    maker = SpectraMaker(A, sightlines)

    # What is the mean of the model?
    def fun(loglam):
        a = 1216. / 10 ** loglam 
        return meanfractionmodel(a)
    meanF = romberg(fun, LogLamMin, LogLamMax) / (LogLamMax - LogLamMin)
    def fun(loglam):
        a = 1216. / 10 ** loglam 
        return varfractionmodel(a)
    varF = romberg(fun, LogLamMin, LogLamMax) / (LogLamMax - LogLamMin)
    stdF = varF ** 0.5

    if False:
        # OK we have collected the tau values now use brentq to 
        # solve for A.
        Nactivesample = sightlines.ActiveSampleEnd - sightlines.ActiveSampleStart
        ActiveSampleOffset = numpy.concatenate([[0], Nactivesample.cumsum()])
        values = sharedmem.empty(ActiveSampleOffset[-1], 'f8')

        print 'Active Nsamples', len(values)

        def work(i):
            sl = slice(ActiveSampleOffset[i], 
                ActiveSampleOffset[i] + Nactivesample[i])
            dreal, a, deltaLN, Dfactor = maker.lognormal(i)
            values[sl] = deltaLN
        chunkmap(work, range(len(sightlines)), 100)

        # now values holds the deltaLNs.

        # we simply divide the optical depth by this factor on every taureal
        # to simulate the effect of splatting.
        # this shall not change the variance in the wrong way.

        N = 1.0 * len(values) / sightlines.Npixels.sum()
        print "samples per pixel", N

        G = numpy.int32(numpy.arange(len(values)) / N)

        def cost(Af, Bf):
            taureal = values ** Bf * A.LogNormalScale
            taureal2 = numpy.bincount(G, weights=taureal)
            F = numpy.exp(-Af * taureal2)
            xmeanF = F.mean()
            xstdF = F.std() 
            v = (xmeanF/ meanF - 1) ,  (xstdF / stdF - 1) 
            return v

    F = sharedmem.empty(sightlines.Npixels.sum(), 'f8')
    F[...] = numpy.nan
    def cost(Af, Bf):
        def work(i):
            sl = slice(sightlines.PixelOffset[i], 
                sightlines.PixelOffset[i] + sightlines.Npixels[i])
            if sightlines.Npixels[i] == 0: return
            taured = maker.convolve(i, 
                    Afunc=lambda x: Af,
                    Bfunc=lambda x: Bf, returns=['taured']).taured
            F[sl] = numpy.exp(-taured)
        chunkmap(work, range(0, len(sightlines), 100), 100)
        F1 = F[~numpy.isnan(F)]
        xmeanF = F1.mean()
        xstdF = F1.std() 
        v = (xmeanF/ meanF - 1) ,  (xstdF / stdF - 1) 
        return v

    r = root(lambda x: cost(*x), (Afguess, Bfguess), method='lm')
    Af, Bf = r.x
    print r.x, r.fun

    cost(Af, Bf) # this will update F
    F1 = F[~numpy.isnan(F)]
    xmeanF = F1.mean()
    xstdF = F1.std()
    print "lam range", 10**LogLamMin, 10**LogLamMax
    print "Af, Bf", Af, Bf
    print 'check', xmeanF, meanF, xstdF, stdF
    return Af, Bf, xmeanF, xstdF

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
