import numpy
import sharedmem
from common import Config
from common import SpectraOutput
from common import Sightlines
from common import FGPAmodel
from lib.chunkmap import chunkmap
import fitsio
from sys import argv
import os

def main(A):
    MatchUp = numpy.load('matchup-result.npy')
    spectra = SpectraOutput(A)
    print MatchUp.dtype
    def work(i):
        match = MatchUp[i]

        dirname = os.path.join(
            'raw', 
            '%04d' % match['PLATE'])
        try:
            os.makedirs(dirname)
        except OSError:
            pass

        print i, '/', len(MatchUp)
        filename = os.path.join(
            dirname,
            'mockraw-%04d-%d-%04d.fits' % 
            (match['PLATE'],
            match['MJD'],
            match['FIBER'])
            )
             
        ind = match['Q_INDEX']

        data = numpy.rec.fromarrays(
                [spectra.z[ind], numpy.exp(-spectra.taured[ind])],
                names=['Z', 'mockF'])

        header = dict(
                DEC=spectra.sightlines.DEC[ind] / numpy.pi * 180,
                RA=spectra.sightlines.RA[ind] / numpy.pi * 180,
                Z=spectra.sightlines.Z_RED[ind])
                
        with fitsio.FITS(filename, mode='rw') as file:
            file.write_table(data, header=header)
    chunkmap(work, range(len(MatchUp)), chunksize=1000)
if __name__ == '__main__':
    main(Config(argv[1]))
