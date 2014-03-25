import numpy
import sharedmem

from common import Config
from sightlines import Sightlines
from lib.lazy import Lazy
from lib.interp1d import  interp1d
from lib.irconvolve import irconvolve
from lib.splat import splat

class SpectraOutput(object):
    def __init__(self, config):
        self.config = config
        sightlines = Sightlines(config)
        class Accessor(object):
            def __init__(self, data):
                self.data = data
            def __getitem__(self, index):
                sl = slice(
                    sightlines.PixelOffset[index],
                    sightlines.PixelOffset[index] + 
                    sightlines.Npixels[index])
                return self.data[sl] 
        self.Accessor = Accessor
        class Faker(object):
            def __init__(self, table):
                self.table = table
            def __getitem__(self, index):
                sl = slice(
                sightlines.LogLamGridIndMin[index],
                sightlines.LogLamGridIndMax[index])
                return self.table[sl]
        self.Faker = Faker
        self.sightlines = sightlines
    def __len__(self):
        return len(self.sightlines)
    @Lazy
    def taured(self):
        taured = numpy.memmap(self.config.SpectraOutputTauRed, mode='r+', dtype='f4')
        return self.Accessor(taured) 
        
    @Lazy
    def taureal(self):
        taureal = numpy.memmap(self.config.SpectraOutputTauReal, mode='r+', dtype='f4')
        return self.Accessor(taureal) 

    @Lazy
    def delta(self):
        delta = numpy.memmap(self.config.SpectraOutputDelta, mode='r+', dtype='f4')
        return self.Accessor(delta)

    @Lazy
    def LogLam(self):
        LogLamGrid = self.config.LogLamGrid
        LogLamCenter = 0.5 * (LogLamGrid[1:] + LogLamGrid[:-1])
        return self.Faker(LogLamCenter)

    @Lazy
    def z(self):
        LogLamGrid = self.config.LogLamGrid
        LogLamCenter = 0.5 * (LogLamGrid[1:] + LogLamGrid[:-1])
        z = 10 ** LogLamCenter / 1216.0 - 1
        return self.Faker(z)
        
    @Lazy
    def Dc(self):
        LogLamGrid = self.config.LogLamGrid
        LogLamCenter = 0.5 * (LogLamGrid[1:] + LogLamGrid[:-1])
        z = 10 ** LogLamCenter / 1216.0 - 1
        a = 1 / (z + 1)
        Dc = self.config.cosmology.Dc(a) * self.config.DH
        return self.Faker(Dc)

class SpectraMaker(object):
    def __init__(self, config, sightlines, Af, Bf):
        self.Af = Af
        self.Bf = Bf
        self.sightlines = sightlines
        self.deltafield = numpy.memmap(config.DeltaField, mode='r', dtype='f4')
        self.velfield = numpy.memmap(config.VelField, mode='r', dtype='f4')
        self.var = numpy.loadtxt(config.datadir + '/gaussian-variance.txt')
        self.config = config
        # to preload these properties.
        self.Dplus = config.cosmology.Dplus
        self.FOmega  = config.cosmology.FOmega
        self.Dc = config.cosmology.Dc
        self.Ea = config.cosmology.Ea


    def convolve(self, i):
        # from Rupert
        # 0.12849 = sqrt( 2 KBT / Mproton) in km/s
        # He didn't have 0.5 in the kernel.
        # we do
        SQRT_KELVIN_TO_KMS = 0.12849 / 2. ** 0.5

        config = self.config
        Dplus = config.cosmology.Dplus
        FOmega  = config.cosmology.FOmega
        Dc = config.cosmology.Dc
        Ea = config.cosmology.Ea
        sightlines = self.sightlines
        deltafield = self.deltafield
        velfield = self.velfield
        var = self.var

        sl1 = slice(sightlines.SampleOffset[i], 
                sightlines.SampleOffset[i] + sightlines.Nsamples[i])

        delta = deltafield[sl1]
        losdisp = velfield[sl1]

        # dreal is in Hubble distance units
        dreal = (sightlines.R1[i] + \
            numpy.arange(sightlines.Nsamples[i]) \
            * config.LogNormalScale) / config.DH

        # redshift distortion
        a = Dc.inv(dreal)
        Af = self.Af(a)
        Bf = self.Bf(a)
        Dfactor = Dplus(a) / Dplus(1.0)
        # rsd is also in Hubble distance units
        rsd = losdisp * FOmega(a) * Dfactor / config.DH
        dred = dreal + rsd

        # thermal broadening
        deltaLN = numpy.exp(Dfactor * delta - (Dfactor ** 2 * var) * 0.5)
        T = config.IGMTemperature * (1 + deltaLN) ** (5. / 3. - 1)
        vtherm = SQRT_KELVIN_TO_KMS * T ** 0.5
        dtherm = vtherm / config.C / (a * Ea(a))

        # tau is proportional to taureal, modulated by a slow
        # function A(z), see LeGoff or Croft
        taureal = Af * numpy.float32(deltaLN ** Bf) * config.LogNormalScale
        taured = numpy.float32(irconvolve(dreal, dred, taureal,
            dtherm))
        assert not numpy.isnan(taured).any()

        loglam = numpy.log10(1216.0 / a)
        LogLamGrid = sightlines.GetPixelLogLamBins(i)

        taureal_pix = splat(loglam, taureal, LogLamGrid)
        taured_pix = splat(loglam, taured, LogLamGrid)
        delta_pix = splat(loglam, 1 + delta, LogLamGrid) / splat(loglam, 1.0, LogLamGrid) - 1

        # redshift distort the quasar position 
        Zqso = sightlines.Z_REAL[i]
        dqso = Dc(1 / (Zqso + 1.0))
        dqso = dqso + numpy.interp(dqso, dreal, rsd)
        Zqso = 1.0 / Dc.inv(dqso) - 1

        return (Zqso, taureal_pix[1:-1], taured_pix[1:-1], delta_pix[1:-1])


def main(A):
    """convolve the tau(mass) field, 
    add in thermal broadening and redshift distortion """


    sightlines = Sightlines(A)
    maker = SpectraMaker(A, sightlines)
   
    Npixels = sightlines.Npixels.sum()

    spectaureal = numpy.memmap(A.SpectraOutputTauReal, mode='w+', 
            dtype='f4', shape=Npixels)
    spectaured = numpy.memmap(A.SpectraOutputTauRed, mode='w+', 
            dtype='f4', shape=Npixels)
    specdelta = numpy.memmap(A.SpectraOutputDelta, mode='w+', 
            dtype='f4', shape=Npixels)

    with sharedmem.MapReduce() as pool:
        def work(i):
            sl2 = slice(sightlines.PixelOffset[i], 
                    sightlines.PixelOffset[i] + sightlines.Npixels[i])
            Z_red, taureal_pix, taured_pix, delta_pix = maker.convolve(i)
            spectaureal[sl2] = taureal_pix
            spectaured[sl2] = taured_pix
            specdelta[sl2] = delta_pix
            sightlines.Z_RED[i] = Z_red

        pool.map(work, range(len(sightlines)))

    spectaureal.flush()
    spectaured.flush()
    specdelta.flush()
    sightlines.Z_RED.flush()

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1])) 
