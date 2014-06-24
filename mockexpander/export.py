import numpy
import sharedmem
from common import Config
from common import Sightlines
from common import FGPAmodel
from common import SpectraOutput

from lib.chunkmap import chunkmap
import fitsio

def expandone(spectra, i):
    mockF = numpy.exp(-spectra.taured[i])
    

def main(A):
    spectra = SpectraOutput(A)
    meanFred = MeanFractionMeasured(A, kind='red')
    meanFreal = MeanFractionMeasured(A, kind='real')

    
    for i in range(len(spectra)):

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1])) 
