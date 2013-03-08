import numpy
import sharedmem
from gaepsi.compiledbase.tree import Tree
from scipy.integrate import simps
from args import pixeldtype
BoxSize = 2000000
def buildtree(P):
  tree = Tree(min=[0, 0, 0], ptp=BoxSize, splitthresh=512, nbits=28)
  tree.build(P['pos'], argsort=lambda self, pos, out: 
          sharedmem.argsort(pos, serialargsort=self.argsort, out=out, merge=tree.merge)
  )
  return tree

def powerfromdelta(delta, BoxSize=BoxSize):
  Dplus = 1.0
  K0 = 2 * numpy.pi / BoxSize
  delta_k = numpy.fft.rfftn(delta) / numpy.prod(numpy.float64(delta.shape))
  i, j, k = numpy.unravel_index(range(len(delta_k.flat)), delta_k.shape)
  i = delta_k.shape[0] // 2 - numpy.abs(delta_k.shape[0] // 2 - i)
  j = delta_k.shape[1] // 2 - numpy.abs(delta_k.shape[1] // 2 - j)
  k = (i ** 2 + j ** 2 + k ** 2) ** 0.5
  dig = numpy.digitize(k, numpy.arange(int(k.max())))
  kpk = k.reshape(delta_k.shape) * (numpy.abs(delta_k) ** 2) * K0 ** -3 * Dplus ** 2
  return numpy.arange(dig.max()+1) * K0, numpy.bincount(dig, weights=kpk.ravel()) / numpy.bincount(dig, weights=k.ravel())

def corr(tree, value, start, end, bins):
  vv = numpy.zeros(len(bins) + 1, dtype='f8')
  N = numpy.zeros(len(bins) + 1, dtype='i8')
#  CoM = tree.centerofmass(value)
#  M = tree.count(value)
#  M /= tree.count()
#  i1, i1nodes = tree.query(None, None, resolution=resolution * bins[-1] * tree.invptp[0])
#  p1 = numpy.concatenate((tree.pos[i1], CoM[i1nodes]), axis=0)
#  v1 = numpy.concatenate((value[i1], M[i1nodes]), axis=0)
  pos = tree.pos
  I = numpy.arange(start, min(end, len(pos)))
  print 'total number of elements ', len(pos)
  with sharedmem.Pool() as pool:
    def work(range):
      vv = numpy.zeros(len(bins) + 1, dtype='f8')
      N = numpy.zeros(len(bins) + 1, dtype='i8')
      i2full, sizes = tree.query(pos[range], 
              (bins[-1], bins[-1], bins[-1]), 
              exclude=bins[0])
      offsets = numpy.concatenate([[0], numpy.cumsum(sizes)[:-1]])
      print sizes, offsets
      for ind, i1 in enumerate(range):
        i2 = i2full[offsets[ind]: offsets[ind] + sizes[ind]]
        p2 = tree.pos[i2]
        p1 = pos[i1]
        d = ((p2 - p1) ** 2).sum(axis=-1) ** 0.5
        v1v2 = value[i1] * value[i2]
        dig = numpy.digitize(d, bins)
        vv[:] = vv[:] + numpy.bincount(dig, weights=v1v2, 
                          minlength=len(bins) + 1)
        N[:] = N[:] + numpy.bincount(dig, 
                          minlength=len(bins) + 1)
      return vv, N
    def reduce(result):
      N[:] = N[:] + result[1]
      vv[:] = vv[:] + result[0]
    pool.map(work, pool.arraysplit(I, chunksize=128), callback=reduce)
  c = (vv / N).view(type=numpy.ndarray)[1:-1]
  r = (bins[:-1] + bins[1:]) * 0.5
  return r, c, N[1:-1]

def corrfrompower(K, P, logscale=False, R=None):
  mask = ~isnan(P) & (K > 0)
  K = K[mask]
  P = P[mask]
  P = P * (2 * numpy.pi) ** 3 # going from GADGET to xiao
  if R is None:
    R = 2 * numpy.pi / K
  if logscale:
    weight = K * numpy.exp(-K**2)
    diff = numpy.log(K)
  else:
    weight = numpy.exp(-K**2)
    diff = K
  XI = [4 * numpy.pi / r * \
      numpy.trapz(P * numpy.sin(K * r) * K * weight, diff) for r in R]
  XI = (2 * numpy.pi) ** -3 * numpy.array(XI)

  return R, XI
