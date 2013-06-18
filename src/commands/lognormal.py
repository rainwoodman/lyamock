import numpy
from args import pixeldtype
from density import lognormal
import sharedmem

def main(A):
  """ log normal transformation and convert to Flux """
  Dplus = A.cosmology.Dplus
  Dplus0 = Dplus(1.0)
  delta = A.P('delta')
  print 'gaussian field', len(delta), 'pixels', 
  var = numpy.loadtxt(A.datadir + '/gaussian-variance.txt')
  std = var ** 0.5
  print 'std=', std
  # pixels shall be over dense
  print 'sample mean =', delta.mean()
  print 'sample std =', delta.var()
  chunksize = 1048576
  Zreal = A.P('Zreal', memmap='r')
  flux = A.P('rawflux', justfile='w')
  for i in range(0, len(delta), chunksize):
      SL = slice(i, i + chunksize)
      a = 1 / (1 + Zreal[SL])
      D = Dplus(a) / Dplus0
      d = (delta[SL]) * D
      d = lognormal(d, std=std * D)
      f = numpy.exp(-numpy.exp(A.beta * d)),
      f = numpy.array(f, dtype=pixeldtype['rawflux'])
      f.tofile(flux)
      flux.flush()

"""
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
"""
