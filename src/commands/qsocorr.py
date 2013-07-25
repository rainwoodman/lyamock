import numpy
from kdcount import correlate
import chealpy
from scipy.stats import kde
import sharedmem
from sys import stdout

from args import sightlinedtype

def getrandom(A):
    NR = 800000
#    R = numpy.random.uniform(size=(NR, 3)) * D.ptp(axis=0)[None, :] \
#            + D.min(axis=0)[None, :]
    realradius = numpy.fromfile(A.datadir + '/QSOcatelog.raw',
            dtype=sightlinedtype)['R']
    R = numpy.empty(NR, ('f4', 3))
    w = A.skymask.mask
    mu = numpy.random.uniform(size=NR) * 2 - 1
    ra = numpy.random.uniform(size=NR) * 2 * numpy.pi
    R[:, 2] = mu
    rho = (1 - mu * mu) ** 0.5
    R[:, 0] = rho * numpy.cos(ra)
    R[:, 1] = rho * numpy.sin(ra)
    w = A.skymask(R)
    R = R[w > 0]
    radius = kde.gaussian_kde(realradius).resample(size=len(R)).ravel()
    R *= radius[:, None]
    print R.max(), len(R)
    return R

def getdata(A):
    raw = numpy.fromfile(A.datadir + '/QSOcatelog.raw', dtype=sightlinedtype)
    ra = raw['RA'] / 180 * numpy.pi
    dec = raw['DEC'] / 180 * numpy.pi
    print ra.min(), ra.max(), dec.min(), dec.max()
    R = numpy.empty((len(raw), 3))
    R[:, 2] = numpy.sin(dec)
    x = numpy.cos(dec)
    R[:, 0] = x * numpy.cos(ra)
    R[:, 1] = x * numpy.sin(ra)
    R[...] *= raw['R'][:, None]
    w = A.skymask(R)
    R = R[ w > 0]
    print 'data', R.max(), len(R)
    return R

def main(A):
    data = correlate.points(getdata(A))
    random = correlate.points(getrandom(A))
    r, DD = correlate.paircount(data, data, correlate.RBins(160000, 20))
    r, DR = correlate.paircount(data, random, correlate.RBins(160000, 20))
    r, RR = correlate.paircount(random, random, correlate.RBins(160000, 20))

    corr = DD / DR - 1, DD / RR - 1, (DD + RR - 2 * DR) / RR

    numpy.savetxt(stdout, zip(r, corr[0], corr[1], corr[2], DD, DR, RR),
            fmt='%g')

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(4, 3), dpi=200)
    ax = figure.add_axes([.1, .1, .85, .85])
    ax.plot(r / 1000, (r / 1000) ** 2 * corr[2], 'o ', label='LS')
    ax.plot(r / 1000, (r / 1000) ** 2 * corr[0], 'o ', label='DDDR')
    ax.plot(r / 1000, (r / 1000) ** 2 * corr[1], 'o ', label='DDRR')
    ax.legend(ncol=3)
    canvas = FigureCanvasAgg(figure)
    figure.savefig('corr.svg')


