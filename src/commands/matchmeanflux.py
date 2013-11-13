import numpy
import sharedmem
from args import pixeldtype, bitmapdtype
from cosmology import interp1d
from scipy.optimize import brentq

def main(A):
    """convolve the tau(mass) field, 
    add in thermal broadening and redshift distortion """
    indexbyz = IndexByDc(A)
    Afactors = numpy.zeros(len(indexbyz), 'f8')
    if len(indexbyz.ind) * 40 < sharedmem.total_memory():
        memmap = None
        print 'using memory'
    else:
        print 'using memmap'
        memmap = 'r'

    dc = indexbyz.dc
    taureal = A.P('taureal', memmap=memmap)
    taured = A.P('taured', memmap=memmap)
    Dc = A.cosmology.Dc
    with sharedmem.MapReduce(np=0) as pool:
        def iterate(i):
            ind = indexbyz[i]
            center = indexbyz.center[i]
            a = Dc.inv(center)
            Left = 0.0
            Right = 1.0
            # invariance:
            # meanflux[Left] > meanflux_model
            # meanflux[Right] < meanflux_model
            taured_sel = taured[ind]
            flux_model = A.FPGAmeanflux(a)

            if len(taured_sel) == 0:
                afac = nan
                flux_model = 0
                flux = 0
            else:
                def f(afac):
                    with sharedmem.MapReduce() as pool:
                        chunksize = 1048576
                        def sum(i):
                            return numpy.exp(-afac * taured_sel[i:i+chunksize]).sum()
                        fluxsum = numpy.sum(pool.map(sum, range(0, len(taured_sel),
                            chunksize)))
                    flux = fluxsum / len(taured_sel)
                    return (flux - flux_model) / flux_model
                a = 0
                b = 0.01
                s = numpy.sign(f(a))
                while numpy.sign(f(b)) == s:
                    b = b * 2
                afac = brentq(f, a, b)
                flux = numpy.exp(-afac * taured_sel).mean()
            return i, afac, flux, flux_model
        def reduce(i, afac, flux, flux_model):
            Afactors[i] = afac
            print i, '/', len(indexbyz), afac, flux, flux_model
        pool.map(iterate, range(len(indexbyz)), reduce=reduce)
    Afactors = numpy.array(zip(indexbyz.center, Afactors))
    Afactors = Afactors[~numpy.isnan(Afactors[:, 1])]
    numpy.savetxt(A.datadir + '/afactors.txt', Afactors)
    
class IndexByDc(object):
    def __init__(self, A):
        self.dc = A.P('dc')
        ind = sharedmem.argsort(self.dc)
        sorted = self.dc[ind]
        step = 300000 / A.DH
        self.bins = numpy.arange(self.dc.min(), 
                self.dc.max() + step,
                step)


        self.center = 0.5 * (self.bins[1:] + self.bins[:-1])
        self.start = sharedmem.searchsorted(sorted, self.bins, side='left')
        self.end = self.start.copy()
        self.end[:-1] = self.start[1:]
        self.end[-1] = len(sorted)
        print self.start, self.end
        self.ind = ind
    def __len__(self):
        return len(self.center)
    def __getitem__(self, i):
        return numpy.sort(self.ind[self.start[i]:self.end[i]])
