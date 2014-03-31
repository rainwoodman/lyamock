import numpy
import sharedmem
import os.path
from kdcount import correlate
from scipy.stats import kde
from sys import stdout

from common import Config
from common import Sightlines
from common import Skymask


def getrandom(A):
    NR = 800000
#    R = numpy.random.uniform(size=(NR, 3)) * D.ptp(axis=0)[None, :] \
#            + D.min(axis=0)[None, :]
    sightlines = Sightlines(A)
    skymask = Skymask(A)
    realradius = sightlines.R
    R = numpy.empty(NR, ('f4', 3))
    mu = numpy.random.uniform(size=NR) * 2 - 1
    ra = numpy.random.uniform(size=NR) * 2 * numpy.pi
    R[:, 2] = mu
    rho = (1 - mu * mu) ** 0.5
    R[:, 0] = rho * numpy.cos(ra)
    R[:, 1] = rho * numpy.sin(ra)
    w = skymask(R)
    R = R[w > 0]
    radius = kde.gaussian_kde(realradius).resample(size=len(R)).ravel()
    R *= radius[:, None]
    print 'random', R.max(), len(R)
    return R

def getqso(A):
    sightlines = Sightlines(A)
    skymask = Skymask(A)
    ra = sightlines.RA / 180 * numpy.pi
    dec = sightlines.DEC / 180 * numpy.pi
    print ra.min(), ra.max(), dec.min(), dec.max()
    R = numpy.empty((len(sightlines), 3))
    R[:, 2] = numpy.sin(dec)
    x = numpy.cos(dec)
    R[:, 0] = x * numpy.cos(ra)
    R[:, 1] = x * numpy.sin(ra)
    R[...] *= sightlines.R[:, None]
    w = skymask(R)
    R = R[ w > 0]
    print 'data', R.max(), len(R)
    return R

def main(A):
    data = correlate.points(getqso(A))
    random = correlate.points(getrandom(A))
    binning = correlate.RBinning(160000, 20)
    DD = correlate.paircount(data, data, binning)
    DR = correlate.paircount(data, random, binning)
    RR = correlate.paircount(random, random, binning)
    r = 1.0 * len(data) / len(random)
    corr = (DD.sum1 + r ** 2 * RR.sum1 - 2 * r * DR.sum1) / (r ** 2 * RR.sum1)
    numpy.savetxt(stdout, zip(DD.centers, corr), fmt='%g')
    r = DD.centers

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(4, 3), dpi=200)
    ax = figure.add_axes([.1, .1, .85, .85])
    ax.plot(r / 1000, (r / 1000) ** 2 * corr, 'o ', label='LS')
    ax.legend()
    canvas = FigureCanvasAgg(figure)
    figure.savefig(os.path.join(A.datadir, 'quasar-corr.svg'))


if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
