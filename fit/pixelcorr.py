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

def makedata(A, Zmin, Zmax, RfLamMin, RfLamMax):
    spectra = SpectraOutput(A)
    meanFred = MeanFractionMeasured(A, real=False)
    meanFreal = MeanFractionMeasured(A, real=True)
    Npixels = spectra.sightlines.Npixels.sum()
    data = sharedmem.empty(Npixels, ('f4', 3))
    DFred, DFreal, Delta = data.T
    pos = sharedmem.empty(Npixels, ('f4', 3))
    mask = sharedmem.empty(Npixels, '?')
    def work(i):
        sl = spectra.Accessor.getslice(i)
        a = spectra.a[i]
        DFred[sl] = numpy.exp(-spectra.taured[i]) / meanFred(a) - 1
        DFreal[sl] = numpy.exp(-spectra.taureal[i]) / meanFreal(a) - 1
        Delta[sl] = spectra.delta[i]
        pos[sl] = spectra.position(i)

        m = spectra.z[i] > Zmin
        m &= spectra.z[i] < Zmax
        m &= spectra.RfLam(i) > RfLamMin
        m &= spectra.RfLam(i) < RfLamMax
        mask[sl] = m
    chunkmap(work, range(len(spectra)), 100)

    return data[mask], pos[mask]

def main(A):
    delta, pos = makedata(A, 
            Zmin=2.0, Zmax=2.2, RfLamMin=1040, RfLamMax=1185)
    delta = delta[::10]
    pos = pos[::10]
    print len(pos)
    data = correlate.field(pos, value=delta)
    DD = correlate.paircount(data, data, correlate.RBinning(160000, 40))
    r = DD.centers
    xi = DD.sum1 / DD.sum2
    print r.shape, xi.shape
    numpy.savez(os.path.join(A.datadir, 'delta-corr1d-both.npz'), r=r, xi=xi)

    figure = Figure(figsize=(4, 5), dpi=200)
    ax = figure.add_subplot(311)
    ax.plot(r / 1000, (r / 1000) ** 2 * xi[0], 'o ', label='$dF$ RSD')
    ax.legend()
    ax = figure.add_subplot(312)
    ax.plot(r / 1000, (r / 1000) ** 2 * xi[1], 'o ', label='$dF$ Real')
    ax.legend()
    ax = figure.add_subplot(313)
    ax.plot(r / 1000, (r / 1000) ** 2 * xi[2], 'o ', label=r'$\delta_m$')
    ax.legend()
    canvas = FigureCanvasAgg(figure)
    figure.savefig(os.path.join(A.datadir, 'delta-corr-both.svg'))

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
