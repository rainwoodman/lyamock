import numpy
import fitsio
import os
from args import bitmapdtype, sightlinedtype
import sharedmem
import StringIO

def deltaF(flux, Z):
    Zbins = numpy.linspace(Z.min(), Z.max(), 200, endpoint=True)
    dig = numpy.digitize(Z, Zbins)
    meanf = numpy.bincount(dig, flux, minlength=len(Zbins+1)) \
            / numpy.bincount(dig, minlength=len(Zbins +1))
    meanf2 = numpy.bincount(dig, flux ** 2, minlength=len(Zbins+1)) \
            / numpy.bincount(dig, minlength=len(Zbins +1))
    
    meanf[numpy.isnan(meanf)] = 1.0
    meanf2[numpy.isnan(meanf2)] = 1.0

    fbar = numpy.interp(Z, Zbins[1:], meanf[1:-1])
    f2bar = numpy.interp(Z, Zbins[1:], meanf2[1:-1])

    dF = flux / fbar - 1
    var = (f2bar / fbar ** 2) - 1
    return dF, 1.0 / var

def deltaFmodel(A, flux, Z):
    a = 1 / (Z + 1)
    meanflux = A.FPGAmeanflux(a)
    dF = flux / meanflux - 1
    return dF, numpy.ones(len(Z))

def main(A):
    """ export """
    bitmap = numpy.memmap(A.datadir + '/bitmap.raw', mode='r', dtype=bitmapdtype)
    sightlines = A.sightlines

    catelog = A.P('QSOcatelog', memmap='r', dtype=sightlinedtype)
    RA = catelog['RA']
    DEC = catelog['DEC']
    Z = catelog['Z_VI']
    numpy.savetxt(A.datadir + '/ROSS-QSO.txt', numpy.array([RA, DEC, Z]).T, fmt='%g %g %g 1 1 ')

    return 
    with file(A.datadir + '/ROSS-forest.txt', 'w') as f:
        pass
    with sharedmem.MapReduce() as pool:
        chunksize=1048576
        def work(i):
            b = bitmap[i:i+chunksize]
            lam = b['lambda']
            mask = lam < 1200
            mask &= ~numpy.isnan(b['flux'])
            b = b[mask]
            dF, weight = deltaFmodel(A, b['flux'], b['Z'])
            objectid = b['objectid']
            DEC = sightlines.DEC[objectid]
            RA = sightlines.RA[objectid]
            Z = b['Z']
            tmpfile = StringIO.StringIO()
            numpy.savetxt(tmpfile,
                        numpy.array([RA, DEC, Z, dF, weight]).T, fmt='%g')
            with pool.critical:
                with file(A.datadir + '/ROSS-forest.txt', 'a') as f:
                    f.write(tmpfile.getvalue())
                print i, len(bitmap)
            tmpfile.close()
        pool.map(work, range(0, len(bitmap), chunksize)) 



