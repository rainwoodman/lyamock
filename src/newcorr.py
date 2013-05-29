import numpy
from args import bitmapdtype
import sharedmem
#sharedmem.set_debug(True)
import kdcount
BoxSize = 12120000.
#BoxSize = 500000.
Npix = 80
LargeScale = 50000.0
bitmap = numpy.fromfile('bitmap.raw', dtype=bitmapdtype)
bitmap = bitmap.reshape(-1, Npix)
bad = numpy.isnan(bitmap['F']).any(axis=-1)
bitmap = bitmap[~bad].reshape(-1).ravel()
#good = (bitmap['Z'] > 2.0) & (bitmap['Z'] < 3.0)
bitmap = bitmap[good]
#good2 = ((bitmap['pos'] - bitmap['pos'][192336]) ** 2).sum(axis=-1) < 2 * 18000.0 ** 2
#bitmap = bitmap[good2]
mubins = numpy.linspace(0.0, 1.0, 10, endpoint=False)
rbins = numpy.linspace(2000, LargeScale, 64, endpoint=False)

C = numpy.zeros((rbins.size + 1, mubins.size + 1) , dtype='i8')
W = numpy.zeros((rbins.size + 1, mubins.size + 1) , dtype='f8')

Ni = numpy.zeros(bitmap.shape, dtype='i8')
Nj = numpy.zeros(bitmap.shape, dtype='i8')

def fastdig(data, bins):
    idx = 1.0 / (bins[1] - bins[0])
    id = numpy.int32((data - bins[0]) * idx) + 1
    id.clip(0, bins.size, out=id)
    return id

def eval_mu(x, y, r):
    dx = y - x
    mid = dx.copy()
    mid *= 0.5
    mid += x
    mid -= BoxSize * 0.5
    dot = numpy.einsum('ij,ij->i', mid, dx)
    dist = numpy.einsum('ij,ij->i', mid, mid) ** 0.5
    dist *= r
    mu = dot / dist
    mu.clip(-1, 1, out=mu)
    return mu

def process(r, i, j, bitmap, C1, W1, Ni1, Nj1):
    good = r > 0
#    good &= (bitmap['row'][i] != bitmap['row'][j])
    i = i[good]
    if len(i) == 0: return

    j = j[good]
    r = r[good]

    x = bitmap['pos'][i]
    y = bitmap['pos'][j]
    
    mu = eval_mu(x, y, r)
#    mu, r = mu * r, r * (1 - mu * mu) ** 0.5
    rdig = fastdig(r, rbins)
    mudig = fastdig(numpy.abs(mu), mubins)
    dig = numpy.ravel_multi_index((rdig, mudig), C.shape)
    C1[:] += numpy.bincount(dig, minlength=C1.size)

    w = bitmap['F'][i] * bitmap['F'][j]
    W1[:] += numpy.bincount(dig, minlength=W1.size, weights=w)
    Ni1[:] += numpy.bincount(i, minlength=Ni1.size)
    Nj1[:] += numpy.bincount(j, minlength=Nj1.size)

def paracount(tree):
    print 'counting', tree.size
    with sharedmem.Pool() as pool:
        def work(other):
            C1 = numpy.zeros_like(C)
            W1 = numpy.zeros_like(W)
            Ni1 = numpy.zeros_like(Ni)
            Nj1 = numpy.zeros_like(Nj)
            tree.enum(other, LargeScale, process, bunch=128 * 1024, 
                    bitmap=bitmap, 
                    C1=C1.ravel(),
                    W1=W1.ravel(), 
                    Ni1=Ni1, Nj1=Nj1)
            return C1, W1, Ni1, Nj1
        def reduce(ret):
            C1, W1, Ni1, Nj1 = ret
            C[:] += C1
            W[:] += W1
            Ni[:] += Ni1
            Nj[:] += Nj1

        pool.map(work, 
                tree.subtrees(tree.size // 512), callback=reduce)

D = kdcount.build(bitmap['pos'])
paracount(D)

numpy.savez('corr.npz', C=C, W=W, mubins=mubins, rbins=rbins)

