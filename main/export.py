import numpy
import sharedmem
from common import Config
from common import Sightlines
from common import FGPAmodel
from common import SpectraOutput

from lib.chunkmap import chunkmap

def main(A):
    sightlines = Sightlines(A)
    fgpa = FGPAmodel(A)

    Npixels = sightlines.Npixels.sum()
    specloglam = numpy.memmap(A.SpectraOutputLogLam, mode='w+', 
            dtype='f4', shape=Npixels)
    # now save LogLam of the pixels for ease of access
    # (not used by our code)
    LogLamGrid = A.LogLamGrid
    LogLamCenter = 0.5 * (LogLamGrid[1:] + LogLamGrid[:-1])
    for index in range(len(sightlines)):
        sl2 = slice(sightlines.PixelOffset[index], 
                sightlines.PixelOffset[index] + sightlines.Npixels[index])
        sl = slice(
            sightlines.LogLamGridIndMin[index],
            sightlines.LogLamGridIndMax[index] - 1)
        specloglam[sl2] = LogLamCenter[sl]
    specloglam.flush()

    # now save QSONpixel for ease of access
    # (not used by our code)
    QSONpixel = numpy.memmap(A.QSONpixel, mode='w+', 
            dtype='i4', shape=len(sightlines))
    QSONpixel[...] = numpy.int32(sightlines.Npixels)
    QSONpixel.flush()

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1])) 
