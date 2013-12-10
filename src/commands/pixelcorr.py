import numpy
from kdcount import correlate
import chealpy
import sharedmem
from sys import stdout
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from args import bitmapdtype
from measuremeanF import meanFmeasured

def deltaFmodel(A, F, Z):
    a = 1 / (Z + 1)
    meanF = A.FPGAmeanF(a)
    dF = F / meanF - 1
    return dF, numpy.ones(len(Z))

def getdata(A):
    meanFm = meanFmeasured('F')
    meanFmreal = meanFmeasured('Freal')

    data = numpy.fromfile(A.datadir + '/bitmap.raw',
            dtype=bitmapdtype)
    mask = (data['Z'] > 2.0) & (data['Z'] < 2.2)
    mask &= (data['lambda'] < 1185)
    mask = ~numpy.isnan(data['F'])
    mask &= ~numpy.isnan(data['Freal'])
    mask &= ~numpy.isnan(data['delta'])
    data = data[mask]
    pos = data['pos'].copy()
    dF = sharedmem.empty(data['F'].shape, 'f8')
    dFreal = sharedmem.empty(data['F'].shape, 'f8')

    with sharedmem.MapReduce() as pool:
        chunksize = 1024 * 1024
        def work(i):
            dF[i:i+chunksize] = data['F'][i:i+chunksize] / meanFm(data['Z'][i:i+chunksize]) - 1
            dFreal[i:i+chunksize] = data['Freal'][i:i+chunksize] / meanFmreal(data['Z'][i:i+chunksize]) - 1
        pool.map(work, range(0, len(data), chunksize))

    delta = numpy.array([
        data['delta'], 
        dF,
        dFreal,
        ]).T.copy()
    print 'F', dF.mean(dtype='f8')
    print 'Freal', dFreal.mean(dtype='f8')
    print 'delta', data['delta'].mean(dtype='f8')
    return pos[::100], delta[::100]

def main(A):
    pos, delta = getdata(A)
    print len(pos)
    data = correlate.field(pos, value=delta)
    DD = correlate.paircount(data, data, correlate.RBinning(160000, 40))
    r = DD.centers
    xi = DD.sum1 / DD.sum2
    print r.shape, xi.shape
    numpy.savez('delta-corr1d-both.npz', r=r, xi=xi)

    figure = Figure(figsize=(4, 5), dpi=200)
    ax = figure.add_subplot(311)
    ax.plot(r / 1000, (r / 1000) ** 2 * xi[0], 'o ', label='d')
    ax.legend()
    ax = figure.add_subplot(312)
    ax.plot(r / 1000, (r / 1000) ** 2 * xi[1], 'o ', label='f')
    ax.legend()
    ax = figure.add_subplot(313)
    ax.plot(r / 1000, (r / 1000) ** 2 * xi[2], 'o ', label='freal')
    ax.legend()
    canvas = FigureCanvasAgg(figure)
    figure.savefig('delta-corr-both.svg')

