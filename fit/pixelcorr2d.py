import numpy
from kdcount import correlate
import sharedmem
import os.path
from common import Config
from sys import stdout

from pixelcorr import makedata

def main(A):
    delta, pos = makedata(A, 
            Zmin=2.0, Zmax=2.2, RfLamMin=1040, RfLamMax=1185)
    delta = delta[::4]
    pos = pos[::4]
    print len(pos)
    print pos.min(), pos.max()
    data = correlate.field(pos, value=delta)
    DD = correlate.paircount(data, data, 
        correlate.RmuBinning(80000, Nbins=20, Nmubins=48, observer=A.BoxSize * 0.5))
    numpy.savez(os.path.join(A.datadir, 'pixcorr-Rmu.npz'), center=DD.centers, sum1=DD.sum1, sum2=DD.sum2)

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
