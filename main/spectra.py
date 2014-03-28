import numpy

from common import Config
from common import Sightlines
from common import FGPAmodel
from lib.lazy import Lazy
from lib.interp1d import  interp1d
from lib.irconvolve import irconvolve
from lib.splat import splat
from lib.chunkmap import chunkmap


class SpectraMaker(object):
    def __init__(self, config, sightlines):
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

    def lognormal(self, i):
        config = self.config
        Dplus = config.cosmology.Dplus
        FOmega  = config.cosmology.FOmega
        Dc = config.cosmology.Dc
        Ea = config.cosmology.Ea
        sightlines = self.sightlines
        deltafield = self.deltafield
        velfield = self.velfield
        var = self.var

        sl1 = slice(sightlines.SampleOffset[i] + 
                sightlines.ActiveSampleStart[i], 
                sightlines.SampleOffset[i] + 
                sightlines.ActiveSampleEnd[i])

        delta = deltafield[sl1]
        # dreal is in Hubble distance units
        dreal = (sightlines.R1[i] + \
            numpy.arange(
                sightlines.ActiveSampleStart[i],
                sightlines.ActiveSampleEnd[i]) \
            * config.LogNormalScale) / config.DH

        # redshift distortion
        a = Dc.inv(dreal)

        Dfactor = Dplus(a) / Dplus(1.0)

        deltaLN = numpy.exp(Dfactor * delta - (Dfactor ** 2 * var) * 0.5)
        return dreal, a, deltaLN, Dfactor

    def convolve(self, i, Afunc, Bfunc, returns=['taured', 'taureal', 'delta', 'Zqso']):
        """
            convolve the ith sightline, with Afunc(a) and Bfunc(a)
            for the A B factors,

            returns taureal and delta of the pixels on the sightline.
            (tuple of 2)

            if withred:
               in addition returns the taured and rsd Z_qso of the sightline.
               (tuple of 4)
        """
        rt = lambda : None
        # from Rupert
        # 0.12849 = sqrt( 2 KBT / Mproton) in km/s
        # He didn't have 0.5 in the kernel.
        # we do
        SQRT_KELVIN_TO_KMS = 0.12849 / 2. ** 0.5

        config = self.config
        FOmega  = config.cosmology.FOmega
        Dc = config.cosmology.Dc
        Ea = config.cosmology.Ea
        sightlines = self.sightlines
        deltafield = self.deltafield
        velfield = self.velfield
        var = self.var

        sl1 = slice(sightlines.SampleOffset[i] + 
                sightlines.ActiveSampleStart[i], 
                sightlines.SampleOffset[i] + 
                sightlines.ActiveSampleEnd[i])

        delta = deltafield[sl1]
        losdisp = velfield[sl1]

        dreal, a, deltaLN, Dfactor = self.lognormal(i)

        Af = Afunc(a)
        Bf = Bfunc(a)

        # tau is proportional to taureal, modulated by a slow
        # function A(z), see LeGoff or Croft
        taureal = Af * deltaLN ** Bf * config.LogNormalScale

        loglam = numpy.log10(1216.0 / a)
        LogLamGrid = sightlines.GetPixelLogLamBins(i)

        T = config.IGMTemperature * (1 + deltaLN) ** (5. / 3. - 1)
        vtherm = SQRT_KELVIN_TO_KMS * T ** 0.5
        dtherm = vtherm / config.C / (a * Ea(a))

        if 'taureal' in returns:
            # thermal broadening in real space 
            taureal_th  = numpy.float32(irconvolve(dreal, dreal, taureal,
                    dtherm))
            taureal_pix = splat(loglam, taureal_th, LogLamGrid)

            rt.taureal = taureal_pix[1:-1]

        if 'delta' in returns:
            w = splat(loglam, 1.0, LogLamGrid)
            w[w == 0] = 1.0
            delta_pix = splat(loglam, 1 + delta, LogLamGrid) / w - 1
            rt.delta = delta_pix[1:-1]

        if 'taured' in returns:
            # rsd is also in Hubble distance units
            rsd = losdisp * FOmega(a) * Dfactor / config.DH
            dred = dreal + rsd
            # thermal broadening in redshift space
            taured = numpy.float32(irconvolve(dreal, dred, taureal,
                dtherm))
            assert not numpy.isnan(taured).any()

            taured_pix = splat(loglam, taured, LogLamGrid)
            rt.taured = taured_pix[1:-1]

        if 'Zqso' in returns:
            # redshift distort the quasar position 
            Zqso = sightlines.Z_REAL[i]
            dqso = Dc(1 / (Zqso + 1.0))
            dqso = dqso + numpy.interp(dqso, dreal, rsd)
            Zqso = 1.0 / Dc.inv(dqso) - 1
            rt.Zqso = Zqso

        return rt


def main(A):
    """convolve the tau(mass) field, 
    add in thermal broadening and redshift distortion """

    sightlines = Sightlines(A)
    maker = SpectraMaker(A, sightlines)
    fgpa = FGPAmodel(A)

    Npixels = sightlines.Npixels.sum()

    spectaureal = numpy.memmap(A.SpectraOutputTauReal, mode='w+', 
            dtype='f4', shape=Npixels)
    spectaured = numpy.memmap(A.SpectraOutputTauRed, mode='w+', 
            dtype='f4', shape=Npixels)
    specdelta = numpy.memmap(A.SpectraOutputDelta, mode='w+', 
            dtype='f4', shape=Npixels)

    def work(i):
        sl2 = slice(sightlines.PixelOffset[i], 
                sightlines.PixelOffset[i] + sightlines.Npixels[i])
        result =  maker.convolve(i, Afunc=fgpa.Afunc, Bfunc=fgpa.Bfunc)
        spectaureal[sl2] = result.taureal
        spectaured[sl2] = result.taured
        specdelta[sl2] = result.delta
        sightlines.Z_RED[i] = result.Zqso
    chunkmap(work, range(len(sightlines)), 100)

    spectaureal.flush()
    spectaured.flush()
    specdelta.flush()
    sightlines.Z_RED.flush()

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1])) 
