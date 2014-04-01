import numpy
from kdcount import correlate
import sharedmem
import os.path
from common import Config
from sys import stdout

from pixelcorr import getforest

def main(A):
    delta, pos, id = getforest(A, 
            Zmin=2.0, Zmax=2.2, RfLamMin=1040, RfLamMax=1185, combine=4)
    print len(pos)
    print pos.min(), pos.max()
    data = correlate.field(pos, value=delta)
    DD = correlate.paircount(data, data, 
        correlate.RmuBinning(80000, Nbins=20, Nmubins=48, observer=0))
    numpy.savez(os.path.join(A.datadir, 'pixcorr-Rmu.npz'), 
        center=DD.centers, sum1=DD.sum1, sum2=DD.sum2)

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
