import numpy
from kdcount import correlate
import chealpy
import sharedmem
from sys import stdout

from args import bitmapdtype
from pixelcorr import getdata

def main(A):
    pos, delta = getdata(A)
    print len(pos)
    print pos.min(), pos.max()
    data = correlate.field(pos, value=delta)
    DD = correlate.paircount(data, data, 
        correlate.RmuBinning(80000, Nbins=20, Nmubins=48, observer=A.BoxSize * 0.5))
    numpy.savez('pixcorr-Rmu.npz', center=DD.centers, sum1=DD.sum1, sum2=DD.sum2)
