import numpy
import fitsio
import os
from args import bitmapdtype, sightlinedtype
import sharedmem
import StringIO

from measuremeanF import meanFmeasured

def deltaFmodel(A, F, Z):
    a = 1 / (Z + 1)
    meanF = A.FPGAmeanF(a)
    dF = F / meanF - 1
    return dF, numpy.ones(len(Z))

def main(A):
    """ export to Ross's format. dF, needs the measured meanF from meanF.npz, 
        run meansuremeanF first
    """

    bitmap = numpy.memmap(A.datadir + '/bitmap.raw', mode='r', dtype=bitmapdtype)
    sightlines = A.sightlines

    meanFm = meanFmeasured('F')

    catelog = A.P('QSOcatelog', memmap='r', dtype=sightlinedtype)
    RA = catelog['RA']
    DEC = catelog['DEC']
    Z = catelog['Z_VI']
    numpy.savetxt(A.datadir + '/ROSS-QSO.txt', numpy.array([RA, DEC, Z]).T, fmt='%g %g %g 1 1 ')

    with file(A.datadir + '/ROSS-forest.txt', 'w') as f:
        pass
    with file(A.datadir + '/ROSS-FFbar.txt', 'w') as f:
        pass
    with file(A.datadir + '/ROSS-FFbar.raw', 'w') as f:
        pass
    with sharedmem.MapReduce() as pool:
        chunksize=1048576
        def work(i):
            b = bitmap[i:i+chunksize]
            lam = b['lambda']
            mask = ~numpy.isnan(b['F'])
            mask &= ~numpy.isnan(b['delta'])
            mask &= ~numpy.isnan(b['Freal'])
            b = b[mask]
            Fbar = meanFm(b['Z'])
            dF = b['F'] / Fbar - 1
            weight = numpy.ones_like(dF)
            objectid = b['objectid']
            DEC = sightlines.DEC[objectid]
            RA = sightlines.RA[objectid]
            Z = b['Z']
            tmpfile = StringIO.StringIO()
            tmparray = numpy.array([RA, DEC, Z, dF, weight]).T
            numpy.savetxt(tmpfile, tmparray, fmt='%g')
            with pool.critical:
                with file(A.datadir + '/ROSS-forest.txt', 'a') as f:
                    f.write(tmpfile.getvalue())
                print i, len(bitmap), dF.mean()

            tmpfile = StringIO.StringIO()
            tmparray = numpy.array([RA, DEC, Z, b['F'], Fbar]).T
            numpy.savetxt(tmpfile, tmparray, fmt='%g')

            with pool.critical:
                with file(A.datadir + '/ROSS-FFbar.txt', 'a') as f:
                    f.write(tmpfile.getvalue())
                with file(A.datadir + '/ROSS-FFbar.raw', 'a') as f:
                    tmparray.tofile(f)
            tmpfile.close()
        pool.map(work, range(0, len(bitmap), chunksize)) 

