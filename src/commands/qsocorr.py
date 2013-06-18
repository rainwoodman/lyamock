import numpy
import kdcount
import chealpy
from scipy.stats import kde
import sharedmem

def corrfun(tree1, tree2, rbins):
    P12 = numpy.zeros(rbins.size + 1, dtype='i8')
    W12 = numpy.zeros((rbins.size + 1, tree1.store.Nw), dtype='f8')
    if tree2.size > tree1.size:
        tree1, tree2 = tree2, tree1
    def work(node1):
        #P12 = numpy.zeros(rbins.size + 1, dtype='i8')
        #for r, i, j in node.enum(tree2, rbins.max(), bunch=100000):
        #    dig = numpy.digitize(r, rbins)
        #    P12 += numpy.bincount(dig, minlength=P12.size)
        #return P12
        return node1.count(tree2, rbins)
    def reduce(P, W):
        P12[1:-1] += numpy.diff(P)
        W12[1:-1] += numpy.diff(W, axis=0)
    sharedmem.Pool(use_threads=False).map(work, 
            tree1.tearoff(50000), reduce=reduce)
    DD = P12[1:-1] * (1.0 * tree1.size * tree2.size) ** -1
    WW = W12[1:-1] * (1.0 * tree1.weight[None, :] * tree2.weight[None, :]) ** -1
    return DD, WW

def main(A):
    global RR, corr, DR, DD, RD, rcenter, Rtree, Dtree, rD, rR
    Nbins = 40
    D = numpy.fromfile(A.datadir + '/QSOcatelog.raw', dtype='f4').reshape(-1,
            7)[:, 4:]
    rD = numpy.fromfile(A.datadir + '/QSOcatelog.raw', dtype='f4').reshape(-1,
            7)[:, 3]
    print len(D)
    NR = 8000000
#    R = numpy.random.uniform(size=(NR, 3)) * D.ptp(axis=0)[None, :] \
#            + D.min(axis=0)[None, :]
    R = numpy.empty(NR, ('f4', 3))
    w = A.skymask.mask
    mu = numpy.random.uniform(size=NR) * 2 - 1
    ra = numpy.random.uniform(size=NR) * 2 * numpy.pi
    R[:, 2] = mu
    rho = (1 - mu * mu) ** 0.5
    R[:, 0] = rho * numpy.cos(ra)
    R[:, 1] = rho * numpy.sin(ra)
    pix = chealpy.vec2pix_nest(A.skymask.Nside, R)
    R = R[w[pix] > 0]
    radius = kde.gaussian_kde(rD).resample(size=len(R)).ravel()
    R *= radius[:, None]
    print rD.max(), rD.min()
    rR = radius
    print 'random size = ', len(R)
#    radius = numpy.random.uniform(rD.min() ** 3, rD.max() ** 3, 
#            size=NR) ** (1 / 3.)
    Rtree = kdcount.build(R)
    Dtree = kdcount.build(D)
    
    rbins = numpy.linspace(10000, 200000, Nbins + 1, endpoint=True)
    rcenter = (rbins[1:] * rbins[:-1]) ** 0.5
    
    DD, junk = corrfun(Dtree, Dtree, rbins)
    print 'DD'
    DR, junk = corrfun(Dtree, Rtree, rbins)
    print 'DR'
    RR, junk = corrfun(Rtree, Rtree, rbins)

    print DD.sum(),  DR.sum(), len(D), len(R) 
    corr = DD / DR - 1, DD / RR - 1, (DD + RR - 2 * DR) / RR
