import numpy
from args import pixeldtype1, pixeldtype2
from args import writefill
import sharedmem

def main(A):
  """ log normal transformation and convert to Flux """
  mean, std = normalization(A)
  Dplus = A.cosmology.Dplus
  Dplus0 = Dplus(1.0)
  def work(i, j, k):
    try:
        fill1 = numpy.fromfile(A.basename(i, j, k, 'pass1') + '.raw',
                pixeldtype1)
    except IOError:
        return
    fill2 = fill1.view(dtype=pixeldtype2)
    a = 1 / (1 + fill1['Z'])
    D = Dplus(a) / Dplus0
    fill1['delta'] = (fill1['delta'] - mean) * D
    lognormal(fill1['delta'], std=std * D, out=fill1['delta'])
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
