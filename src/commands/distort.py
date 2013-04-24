import numpy
from args import pixeldtype, pixeldtype2
from args import writefill
import sharedmem

def main(A):
  """ log normal transformation and redshift distortion"""
  mean, std = normalization(A)
  Dplus = A.cosmology.Dplus
  FOmega = A.cosmology.FOmega
  Ea = A.cosmology.Ea
  H0 = 0.1
  Dplus0 = Dplus(1.0)
  def work(i, j, k):
    try:
        fill1 = numpy.fromfile(A.basename(i, j, k, 'pass1') + '.raw',
                pixeldtype)
    except IOError:
        return
    fill2 = fill1.view(dtype=pixeldtype2)
    a = 1 / (1 + fill1['Z'])
    D = Dplus(a) / Dplus0
    fill1['delta'] = (fill1['delta'] - mean) * D
    lognormal(fill1['delta'], std=std * D, out=fill1['delta'])
    losvel = a * D * FOmega(a) * Ea(a) * H0 * fill1['losdisp']
    print 'losvel stat', losvel.max(), losvel.min()
    # moving away is positive
    dz = - losvel / A.C
    fill2['Zshift'] = (1 + dz) * (1 + fill2['Z0']) - 1 - fill2['Z0']
    print 'Z0', fill2['Z0'].max(), fill2['Z0'].min()
    print 'Zshift', fill2['Zshift'].max(), fill2['Zshift'].min()
    fill2['F'] = numpy.exp(-numpy.exp(A.beta * fill1['delta']))
    writefill(fill2, A.basename(i, j, k, 'pass2')) 

  with sharedmem.Pool() as pool:
    pool.starmap(work, list(A.yieldwork()))

def normalization(A):
  K1sum = 0.0
  K2sum = 0.0
  Nsum = 0
  for i, j, k in A.yieldwork():
    G = {}
    try:
        execfile(A.basename(i, j, k, 'pass1') + '.info-file', G)
        K1sum += G['K']
        K2sum += G['K2']
        Nsum += G['N']
    except IOError:
        pass
  mean = K1sum / Nsum
  std = (K2sum / Nsum - mean ** 2) ** 0.5
  print 'mean is', mean, 'std is', std
  return mean, std
