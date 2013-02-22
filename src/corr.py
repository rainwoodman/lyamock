import numpy
import sharedmem
from gaepsi.compiledbase.tree import Tree
from scipy.integrate import simps
pixeldtype = numpy.dtype(
    [ ('pos', ('f4', 3)), ('row', 'i4'), ('ind', 'i4'), ('Z', 'f4')])

datadtype = numpy.dtype([('delta', 'f4'), ('disp', 'f4')])

def buildtree(P):
  tree = Tree(P['pos'], min=[0, 0, 0], ptp=12000000, splitthresh=256, nbits=28)
  return tree

def corr(pos, tree, value, Nsample, bins):
  assert(len(pos) == len(value))
  I = numpy.random.randint(len(pos), size=Nsample)
  vv = sharedmem.empty(len(bins) + 1, dtype='f8')
  N = sharedmem.empty(len(bins) + 1, dtype='i8')
  vv[:] = 0
  N[:] = 0
  with sharedmem.Pool() as pool:
    def work(i):
      res = tree.query(pos[i], (bins[-1], bins[-1], bins[-1]), exclude=bins[0])
      if len(res) == 0: return
      res2 = tree.query(pos[i], (bins[-1], bins[-1], bins[-1]))
      dx = numpy.abs(pos[res] - pos[i])
      d = ((dx) ** 2).sum(axis=-1) ** 0.5
      dx2 = numpy.abs(pos[res2] - pos[i])
      d2 = ((dx2) ** 2).sum(axis=-1) ** 0.5
      print len(res), len(res2),
      print (d < bins[0]).sum(), (d2 < bins[0]).sum()
      dig = numpy.digitize(d, bins)
      vv[:] = vv[:] + numpy.bincount(dig, weights=value[i] * value[res], minlength=len(bins) + 1)
      N[:] = N[:] + numpy.bincount(dig, minlength=len(bins) + 1)
    pool.map(work, I)
  c = (vv / N).view(type=numpy.ndarray)[1:-1]
  r = (bins[:-1] + bins[1:]) * 0.5
  return r, c

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
