import numpy
from gaepsi.cosmology import WMAP7
import sharedmem

BOX = 12000000.
N = 32
ZCUT = 1.5
LINES = numpy.loadtxt('../DR10Q_alpha_short.dat', usecols=(4, 5, 6, 0), dtype=[('RA', 'f4'), ('DEC', 'f4'), ('Z_VI', 'f4'), ('THINGID', 'i8')])
RA, DEC, Z_VI, THINGID = LINES['RA'], LINES['DEC'], LINES['Z_VI'], LINES['THINGID']

pixeldtype = numpy.dtype(
    [ ('pos', ('f4', 3)), ('row', 'i4'), ('ind', 'i4'), ('Z', 'f4')])

lambdagrid = 10 ** numpy.arange(numpy.log10(1026), numpy.log10(1216), 0.0001)
Z = (1 + Z_VI[None, :]) * lambdagrid[:, None] / 1216 - 1
print Z.shape
P = sharedmem.empty(shape=Z.shape, dtype=pixeldtype)
with sharedmem.Pool() as pool:
  def work(i):
    P[i]['row'] = numpy.arange(len(RA))[None, :]
    P[i]['ind'] = i
    P[i]['pos'] = WMAP7.radec2pos(ra=RA / 180 * numpy.pi, 
                             dec=DEC / 180 * numpy.pi,
                             z=Z[i]) + BOX * 0.5
    P[i]['Z'] = Z[i]
    
  pool.map(work, range(len(lambdagrid)))

P = P.ravel()
P = P[(P['Z'] > ZCUT)]
print P.shape

def work(i):
  subset_i = P[(P['pos'][:, 0] >= i * BOX / N) & (P['pos'][:, 0] < (i + 1) * BOX / N)]
  for j in range(N):
    subset_j = subset_i[(subset_i['pos'][:, 1] >= j * BOX / N) & (subset_i['pos'][:, 1] < (j + 1) * BOX / N)]
    for k in range(N):
      Subset = subset_j[(subset_j['pos'][:, 2] >= k * BOX / N) & (subset_j['pos'][:, 2] < (k + 1) * BOX / N)]
      if len(Subset) > 0:
        print Subset.shape
        Subset.tofile('%d/%02d-%02d-%02d-pixels.raw' % (N, i, j, k))

with sharedmem.Pool(use_threads=False) as pool:
  pool.map(work, range(N))
