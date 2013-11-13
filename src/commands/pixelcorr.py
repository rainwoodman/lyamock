import numpy
from kdcount import correlate
import chealpy
import sharedmem
from sys import stdout
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from args import bitmapdtype

def deltaF(flux, Z):
    Zbins = numpy.linspace(Z.min(), Z.max(), 200, endpoint=True)
    dig = numpy.digitize(Z, Zbins)
    meanf = numpy.bincount(dig, flux, minlength=len(Zbins+1)) \
            / numpy.bincount(dig, minlength=len(Zbins +1))
    
    meanf[numpy.isnan(meanf)] = 1.0
    fbar = numpy.interp(Z, Zbins[1:], meanf[1:-1])
    return flux / fbar - 1

def deltaFmodel(A, flux, Z):
    a = 1 / (Z + 1)
    meanflux = A.FPGAmeanflux(a)
    dF = flux / meanflux - 1
    return dF, numpy.ones(len(Z))

def meanflux(vars, Z):
    Zbins = numpy.linspace(Z.min(), Z.max(), 200, endpoint=True)
    dig = numpy.digitize(Z, Zbins)
    meanf = [
        numpy.bincount(dig, flux, minlength=len(Zbins+1)) \
            / numpy.bincount(dig, minlength=len(Zbins +1))
            for flux in vars
            ]
    return Zbins[1:], [ i[1:-1] for i in meanf]

def getdata(A):
    data = numpy.fromfile(A.datadir + '/bitmap.raw',
            dtype=bitmapdtype)
    #mask = (data['Z'] > 2.0) & (data['Z'] < 2.2)
    #mask &= (data['lambda'] < 1185)
    mask = ~numpy.isnan(data['flux'])
    mask &= ~numpy.isnan(data['fluxreal'])
    mask &= ~numpy.isnan(data['delta'])
    data = data[mask]

    z, meanf = meanflux(
            (data['delta'], data['flux'], data['fluxreal']),
            data['Z'])
    delta, flux, fluxreal = meanf

    numpy.savez('meanflux.npz', z=z, delta=delta, flux=flux, fluxreal=fluxreal)
    
    figure = Figure(figsize=(4, 5), dpi=200)
    ax = figure.add_subplot(311)
    ax.plot(z, delta, 'o ', label='d')
    ax.legend()
    ax = figure.add_subplot(312)
    ax.plot(z, flux, '. ', label='d')
    ax.plot(z, A.FPGAmeanflux(1 / (z + 1)), '-', label='model')
    ax.legend()
    ax = figure.add_subplot(313)
    ax.plot(z, fluxreal, '. ', label='d')
    ax.plot(z, A.FPGAmeanflux(1 / (z + 1)), '-',
            label='model')
    ax.legend()
    canvas = FigureCanvasAgg(figure)
    figure.savefig('meanflux.svg')

    pos = data['pos'].copy()
    data['flux'] = deltaFmodel(A, data['flux'], data['Z'])
    data['fluxreal'] = deltaFmodel(A, data['fluxreal'], data['Z'])
    delta = numpy.array([
        data['delta'], 
        data['flux'],
        data['fluxreal'],
        ]).T.copy()
    print 'flux', data['flux'].mean()
    print 'delta', data['delta'].mean()

    return pos[::4], delta[::4]

def main(A):
    pos, delta = getdata(A)
    print len(pos)
    data = correlate.field(pos, value=delta)
    xi, bins = correlate.paircount(data, data, correlate.RBins(160000, 40))
     
    r = bins.centers
    print r.shape, xi.shape
    numpy.savez('delta-corr1d-both.npz', r=r, xi=xi)

    figure = Figure(figsize=(4, 5), dpi=200)
    ax = figure.add_subplot(311)
    ax.plot(r / 1000, (r / 1000) ** 2 * xi[0][1:-1], 'o ', label='d')
    ax.legend()
    ax = figure.add_subplot(312)
    ax.plot(r / 1000, (r / 1000) ** 2 * xi[1][1:-1], 'o ', label='f')
    ax.legend()
    ax = figure.add_subplot(313)
    ax.plot(r / 1000, (r / 1000) ** 2 * xi[2][1:-1], 'o ', label='freal')
    ax.legend()
    canvas = FigureCanvasAgg(figure)
    figure.savefig('delta-corr-both.svg')

