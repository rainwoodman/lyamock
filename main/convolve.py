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

def main(A):
    """convolve the tau(mass) field, 
    add in thermal broadening and redshift distortion """
    global sightlines
    global spectra
    global velfield
    global deltafield

    # from Rupert
    # 0.12849 = sqrt( 2 KBT / Mproton) in km/s
    # He didn't have 0.5 in the kernel.
    # we do

    SQRT_KELVIN_TO_KMS = 0.12849 / 2. ** 0.5

    sightlines = Sightlines(A)
    Npixels = sightlines.Npixels.sum()

    spectaureal = numpy.memmap(A.SpectraOutputTauReal, mode='w+', 
            dtype='f4', shape=Npixels)
    spectaured = numpy.memmap(A.SpectraOutputTauRed, mode='w+', 
            dtype='f4', shape=Npixels)
    specdelta = numpy.memmap(A.SpectraOutputDelta, mode='w+', 
            dtype='f4', shape=Npixels)

    var = numpy.loadtxt(A.datadir + '/gaussian-variance.txt')
    print var

    velfield = numpy.memmap(A.VelField, mode='r', dtype='f4')
    deltafield = numpy.memmap(A.DeltaField, mode='r', dtype='f4')

    Dplus = A.cosmology.Dplus
    FOmega  = A.cosmology.FOmega
    Dc = A.cosmology.Dc
    Ea = A.cosmology.Ea
   
    with sharedmem.MapReduce() as pool:
        def convolve(i):
            sl1 = slice(sightlines.SampleOffset[i], 
                    sightlines.SampleOffset[i] + sightlines.Nsamples[i])
            sl2 = slice(sightlines.PixelOffset[i], 
                    sightlines.PixelOffset[i] + sightlines.Npixels[i])

            # assert Npixels and LogLamGridInd are consistent!
            # offset 1 here is to neglect the first pixel 
            sl3 = slice(1 + sightlines.LogLamGridIndMin[i],
                        1 + sightlines.LogLamGridIndMax[i])

            delta = deltafield[sl1]
            losdisp = velfield[sl1]

            # dreal is in Hubble distance units
            dreal = sightlines.R1[i] + \
                numpy.arange(sightlines.Nsamples[i]) \
                * A.LogNormalScale / A.DH

            # redshift distortion
            a = Dc.inv(dreal)
            Dfactor = Dplus(a) / Dplus(1.0)
            # rsd is also in Hubble distance units
            rsd = losdisp * FOmega(a) * Dfactor / A.DH
            dred = dreal + rsd

            # thermal broadening
            deltaLN = numpy.exp(Dfactor * delta - (Dfactor ** 2 * var) * 0.5)
            T = A.IGMTemperature * (1 + deltaLN) ** (5. / 3. - 1)
            vtherm = SQRT_KELVIN_TO_KMS * T ** 0.5
            dtherm = vtherm / A.C / (a * Ea(a))

            # tau is proportional to taureal, modulated by a slow
            # function A(z), see LeGoff or Croft
            taureal = numpy.float32(deltaLN ** A.Beta) * A.LogNormalScale
            taured = numpy.float32(irconvolve(dreal, dred, taureal,
                dtherm))
            assert not numpy.isnan(taured).any()

            loglam = numpy.log10(1216.0 / a)

            taureal_pix = splat(loglam, taureal, A.LogLamGrid)
            taured_pix = splat(loglam, taured, A.LogLamGrid)
            delta_pix = splat(loglam, 1 + delta, A.LogLamGrid) / splat(loglam, 1.0, A.LogLamGrid) - 1

            spectaureal[sl2] = taureal_pix[sl3]
            spectaured[sl2] = taured_pix[sl3]
            specdelta[sl2] = delta_pix[sl3]

            # redshift distort the quasar position 
            Zqso = sightlines.Z_REAL[i]
            dqso = Dc(1 / (Zqso + 1.0))
            dqso = dqso + numpy.interp(dqso, dreal, rsd)
            sightlines.Z_RED[i] = 1.0 / Dc.inv(dqso) - 1

        pool.map(convolve, range(len(sightlines)))
    spectaureal.flush()
    spectaured.flush()
    specdelta.flush()
    sightlines.Z_RED.flush()

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1])) 
