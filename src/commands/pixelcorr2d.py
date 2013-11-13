import numpy
from kdcount import correlate
import chealpy
import sharedmem
from sys import stdout

from args import bitmapdtype
from pixelcorr import deltaF

def getdata(A):
    data = numpy.fromfile(A.datadir + '/bitmap.raw',
            dtype=bitmapdtype)
    mask = (data['Z'] > 2.0) & (data['Z'] < 2.5)
    pos = data['pos'].copy()
    mask = data['lambda'] < 1140
    data['flux'] = deltaF(data['flux'], data['Z'])
    data['fluxred'] = deltaF(data['fluxred'], data['Z'])
    delta = numpy.array([
        data['delta'], 
        data['deltared'],
        data['flux'],
        data['fluxred'],
        ]).T.copy()
    return pos[mask][::2], delta[mask][::2]

def main(A):
    pos, delta = getdata(A)
    print len(pos)
    print pos.min(), pos.max()
    data = correlate.field(pos, value=delta)
    xy, xi = correlate.paircount(data, data, correlate.XYBins(150000, 50,
        A.BoxSize * 0.5))
    numpy.savez('delta-corr2d-both.npz', xy=xy, xi=xi)
