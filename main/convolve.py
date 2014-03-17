import numpy
import sharedmem
from args import Config
from sightlines import Sightlines
from lib.interp1d import  interp1d
from lib.irconvolve import irconvolve
from lib.splat import splat

spectradtype = numpy.dtype([
    ('delta', 'f4'), 
    ('taured', 'f4'), 
    ('taureal', 'f4'), 
    ])

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
    spectra = numpy.memmap(A.SpectraOutput, mode='w+', 
            dtype=spectradtype, 
            shape=sightlines.Npixels.sum())

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
            sl3 = slice(sightlines.LogLamGridIndMin[i],
                        sightlines.LogLamGridIndMax[i])

            delta = deltafield[sl1]
            losdisp = velfield[sl1]
            thisspectra = spectra[sl2]

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

            thisspectra['taureal'] = taureal_pix[sl3]
            thisspectra['taured'] = taured_pix[sl3]
            thisspectra['delta'] = delta_pix[sl3]

            # redshift distort the quasar position 
            Zqso = sightlines.Z_REAL[i]
            dqso = Dc(1 / (Zqso + 1.0))
            dqso = dqso + numpy.interp(dqso, dreal, rsd)
            sightlines.Z_RED[i] = 1.0 / Dc.inv(dqso) - 1

        pool.map(convolve, range(len(sightlines)))
    spectra.flush()
    sightlines.Z_RED.flush()

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1])) 
