import numpy
import os.path
from kdcount import correlate
import sharedmem
from sys import stdout
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from common import Config
from common import SpectraOutput
from common import MeanFractionMeasured
from lib.chunkmap import chunkmap

def getforest(A, Zmin, Zmax, RfLamMin, RfLamMax, combine=1):
    spectra = SpectraOutput(A)
    meanFred = MeanFractionMeasured(A, kind='red')
    meanFreal = MeanFractionMeasured(A, kind='real')

    combine = numpy.minimum(spectra.sightlines.Npixels.max(), combine)

    # will combine every this many pixels
    Npixels1 = spectra.sightlines.Npixels // combine
    Offset1 = numpy.concatenate([[0], numpy.cumsum(Npixels1)])
    Npixels = Npixels1.sum()
    print Npixels1.min(), Npixels1.max()
    print spectra.sightlines.Npixels.min(), spectra.sightlines.Npixels.max()
    data = sharedmem.empty(Npixels, ('f4', 3))
    DFred, DFreal, Delta = data.T
    pos = sharedmem.empty(Npixels, ('f4', 3))
    x, y, z = pos.T
    mask = sharedmem.empty(Npixels, '?')
    id = sharedmem.empty(Npixels, 'i4')

    def work(i):
        def combinepixels(value, method=numpy.mean):
            # reduce the number of pixels with 'method'
            return \
                method(value[:Npixels1[i] * combine]\
                .reshape([Npixels1[i]] + [combine]), 
                    axis=-1)
        sl = slice(Offset1[i], Npixels1[i] + Offset1[i])
        a = spectra.a[i]
        Fred = numpy.exp(-spectra.taured[i]) / meanFred(a) - 1
        Freal = numpy.exp(-spectra.taureal[i]) / meanFreal(a) - 1

        DFred[sl] = combinepixels(Fred)
        DFreal[sl] = combinepixels(Freal)
        Delta[sl] = combinepixels(spectra.delta[i])
        p = spectra.position(i)
        x[sl] = combinepixels(p[:, 0])
        y[sl] = combinepixels(p[:, 1])
        z[sl] = combinepixels(p[:, 2])

        m = spectra.z[i] > Zmin
        m &= spectra.z[i] < Zmax
        m &= spectra.RfLam(i) > RfLamMin
        m &= spectra.RfLam(i) < RfLamMax
        mask[sl] = combinepixels(m, method=numpy.all)
        id[sl] = i
    chunkmap(work, range(len(spectra)), 100)

    return data[mask], pos[mask], id[mask]

def main(A):
    delta, pos, id = getforest(A, 
            Zmin=2.0, Zmax=2.2, RfLamMin=1040, RfLamMax=1185, combine=4)
    print len(pos)
    print pos, delta
    data = correlate.field(pos, value=delta)
    DD = correlate.paircount(data, data, correlate.RBinning(160000, 40))
    r = DD.centers
    xi = DD.sum1 / DD.sum2
    print r.shape, xi.shape
    numpy.savez(os.path.join(A.datadir, 'delta-corr1d-both.npz'), r=r, xi=xi)

    figure = Figure(figsize=(4, 5), dpi=200)
    ax = figure.add_subplot(311)
    ax.plot(r / 1000, (r / 1000) ** 2 * xi[0], 'o ', label='$dF$ RSD')
    ax.set_ylim(-0.4, 1.0)
    ax.legend()
    ax = figure.add_subplot(312)
    ax.plot(r / 1000, (r / 1000) ** 2 * xi[1], 'o ', label='$dF$ Real')
    ax.set_ylim(-0.4, 1.0)
    ax.legend()
    ax = figure.add_subplot(313)
    ax.plot(r / 1000, (r / 1000) ** 2 * xi[2], 'o ', label=r'$dF$ Broadband')
    ax.set_ylim(-20, 60)
    ax.legend()
    canvas = FigureCanvasAgg(figure)
    figure.savefig(os.path.join(A.datadir, 'delta-corr-both.svg'))

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
