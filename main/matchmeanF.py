import numpy
import sharedmem
from scipy.optimize import brentq, fsolve
from scipy.integrate import romberg

from common import Config
from lib.lazy import Lazy
from common import MeanFractionModel
from common import VarFractionModel
from sightlines import Sightlines
from convolve import SpectraMaker

# turns out Af is exp(u * a + v), 
# and Bf is u * a **2 + v * a + w.
# thus we use polyfit in FGPAmodel
from numpy import polyfit, polyval

class FGPAmodel(object):
    def __init__(self, config):
        f = numpy.load(config.MatchMeanFractionOutput)
        a = f['a']
        Af = f['Af']
        Bf = f['Bf']
        arg = a.argsort()
        Af = Af[arg]
        Bf = Bf[arg]
        a = a[arg]
        # reject bad fits
        mask = (Af > 0)
        self.a = a[mask]
        self.Af = Af[mask]
        self.Bf= Bf[mask]
        
    @Lazy
    def Afunc(self):
        pol = polyfit(self.a, numpy.log(self.Af), 1)
        def func(a):
            return numpy.exp(polyval(pol, a))
        return func
    @Lazy
    def Bfunc(self):
        pol = polyfit(self.a, self.Bf, 2)
        def func(a):
            return polyval(pol, a)
        return func

def main(A):
    """ match the mean fraction by fixing prefactor A(a) on tau
        add in thermal broadening and redshift distortion """
    global meanfractionmodel
    global varfractionmodel

    varfractionmodel = VarFractionModel(A)
    meanfractionmodel = MeanFractionModel(A)

    Nbins = 24
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
    numpy.savez(config.MatchMeanFractionOutput, a=1 / (z+1), 
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


    # OK we have collected the tau values now use brentq to 
    # solve for A.
    Nactivesample = sightlines.ActiveSampleEnd - sightlines.ActiveSampleStart
    ActiveSampleOffset = numpy.concatenate([[0], Nactivesample.cumsum()])
    values = sharedmem.empty(ActiveSampleOffset[-1], 'f8')

    print 'Active Nsamples', len(values)

    with sharedmem.MapReduce() as pool:
        chunksize = 100
        def work(j):
            for i in range(j, j + 100):
                if i >= len(sightlines): break 
                sl = slice(ActiveSampleOffset[i], 
                    ActiveSampleOffset[i] + Nactivesample[i])
                dreal, a, deltaLN, Dfactor = maker.lognormal(i)
                values[sl] = deltaLN
        pool.map(work, range(0, len(sightlines), chunksize))

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
    x0 = fsolve(lambda x: cost(*x), (Afguess, Bfguess))
    Af, Bf = x0

    values = sharedmem.empty(sightlines.Npixels.sum(), 'f8')
    # now lets check how good this actually is:
    with sharedmem.MapReduce() as pool:
        chunksize = 100
        def work(j):
            for i in range(j, j + 100):
                if i >= len(sightlines): break
                if sightlines.Npixels[i] == 0: continue 
                sl = slice(sightlines.PixelOffset[i], 
                    sightlines.PixelOffset[i] + sightlines.Npixels[i])
                taureal, delta, taured, Zqso = maker.convolve(i, withrsd=True, Afunc=lambda x: Af,
                        Bfunc=lambda x: Bf)
                values[sl] = taured
        pool.map(work, range(0, len(sightlines), chunksize))
    F = numpy.exp(-values)
    xmeanF = F.mean()
    xstdF = F.std()
    print "lam range", 10**LogLamMin, 10**LogLamMax
    print "Af, Bf", Af, Bf
    print 'check', xmeanF, meanF, xstdF, stdF
    return Af, Bf, xmeanF, xstdF

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
