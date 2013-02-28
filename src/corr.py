import numpy
import sharedmem
from gaepsi.compiledbase.tree import Tree
from scipy.integrate import simps
pixeldtype = numpy.dtype(
    [ ('pos', ('f4', 3)), ('row', 'i4'), ('ind', 'i4'), ('Z', 'f4')])

datadtype = numpy.dtype([('delta', 'f4'), ('disp', 'f4')])
#sharedmem.set_debug(True)

def buildtree(P):
  tree = Tree(min=[0, 0, 0], ptp=12000000, splitthresh=64, nbits=28)
  tree.build(P['pos'], argsort=lambda self, pos, out: 
          sharedmem.argsort(pos, serialargsort=self.argsort, out=out, merge=tree.merge)
  )
  return tree

def corr(tree, value, start, end, bins):
  vv = numpy.zeros(len(bins) + 1, dtype='f8')
  N = numpy.zeros(len(bins) + 1, dtype='i8')
#  CoM = tree.centerofmass(value)
#  M = tree.count(value)
#  M /= tree.count()
#  i1, i1nodes = tree.query(None, None, resolution=resolution * bins[-1] * tree.invptp[0])
#  p1 = numpy.concatenate((tree.pos[i1], CoM[i1nodes]), axis=0)
#  v1 = numpy.concatenate((value[i1], M[i1nodes]), axis=0)
  p1 = tree.pos
  v1 = value
  I = numpy.arange(start, min(end, len(p1)))
  print 'total number of resolved nodes', len(p1)
  with sharedmem.Pool() as pool:
    def work(range):
      vv = numpy.zeros(len(bins) + 1, dtype='f8')
      N = numpy.zeros(len(bins) + 1, dtype='i8')
      for i in range:
        i2 = tree.query(p1[i], (bins[-1], bins[-1], bins[-1]), exclude=bins[0], resolution=None)
        if len(i2) == 0: return
        p2 = tree.pos[i2]
        dx = p2 - p1[i]
        d = ((dx) ** 2).sum(axis=-1) ** 0.5
        v2 = value[i2]
        dig = numpy.digitize(d, bins)
        vv[:] = vv[:] + v1[i] * numpy.bincount(dig, weights=v2, minlength=len(bins) + 1)
        N[:] = N[:] + numpy.bincount(dig, minlength=len(bins) + 1)
      return vv, N
    def reduce(result):
      N[:] = N[:] + result[1]
      vv[:] = vv[:] + result[0]
    pool.map(work, pool.arraysplit(I), callback=reduce)
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
