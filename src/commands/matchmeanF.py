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
            # meanF[Left] > meanF_model
            # meanF[Right] < meanF_model
            taured_sel = taured[ind]
            F_model = A.FPGAmeanF(a)

            if len(taured_sel) == 0:
                afac = numpy.nan
                F_model = 0
                F = 0
            else:
                def f(afac):
                    with sharedmem.MapReduce() as pool:
                        chunksize = 1048576
                        def sum(i):
                            return numpy.exp(-afac * taured_sel[i:i+chunksize]).sum()
                        Fsum = numpy.sum(pool.map(sum, range(0, len(taured_sel),
                            chunksize)))
                    F = Fsum / len(taured_sel)
                    return (F - F_model) / F_model
                a = 0
                b = 0.01
                s = numpy.sign(f(a))
                while numpy.sign(f(b)) == s:
                    b = b * 2
                afac = brentq(f, a, b)
                F = numpy.exp(-afac * taured_sel).mean()
            return i, afac, F, F_model
        def reduce(i, afac, F, F_model):
            Afactors[i] = afac
            print i, '/', len(indexbyz), afac, F, F_model
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
