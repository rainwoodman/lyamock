import numpy
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import map_coordinates
import sharedmem
from sys import argv

pixeldtype = numpy.dtype(
    [ ('pos', ('f4', 3)), ('row', 'i4'), ('ind', 'i4'), ('Z', 'f4')])

def loadpixel(i, j, k, prefix='32/'):
  try:
    return numpy.fromfile(prefix + "%02d-%02d-%02d-pixels.raw" % (i, j, k), dtype=pixeldtype)
  except IOError:
    return numpy.array([], dtype=pixeldtype)


numpy.random.seed(1811720)
Box = 12000000
#PowerSpec = loadpower('power-16384.txt')
PowerSpec = loadpower()
NmeshEff = 8192
Nsample = 256 

if len(argv) == 1: raise Exception("will not run")

Nrep = NmeshEff / Nsample
seed0 = numpy.random.randint(1<<21 - 1)
seedtable = numpy.random.randint(1<<21 - 1, size=(Nrep, Nrep, Nrep))
delta0 = realize(PowerSpec, seed0, Nsample, Nsample, Box)
losdisp0 = realize_losdisp(PowerSpec, [0, 0, 0],
               seed0, Nsample, Nsample, Box)

def dopixels(i, j, k):
  pixels = loadpixel(i, j, k, prefix='%d/' % Nrep)
  if len(pixels) == 0: return numpy.array([])
  cornerpos = [i * Box / Nrep, j * Box / Nrep, k * Box / Nrep]
  delta1 = realize(PowerSpec, seedtable[i, j, k], NmeshEff, 
               Nsample, Box, Nsample)
  losdisp1 = realize_losdisp(PowerSpec, cornerpos,
               seedtable[i, j, k], NmeshEff,
               Nsample, Box, Nsample)
  coarse = pixels['pos'] / Box * Nsample
  fine = (pixels['pos'] / Box * Nrep - [i, j, k]) * Nsample
  d0 = map_coordinates(delta0, coarse.T, mode='wrap')
  disp0 = map_coordinates(losdisp0, coarse.T, mode='wrap')
  d1 = map_coordinates(delta1, fine.T, mode='wrap')
  disp1 = map_coordinates(losdisp1, fine.T, mode='wrap')
  return numpy.array([d1 + d0, disp0 + disp1]).T

if __name__ == '__main__':
  i = int(argv[1])
  def work(jk):
    j = jk // Nrep
    k = jk % Nrep
    delta = dopixels(i, j, k)
    if len(delta) == 0: return
    print i, j, k, delta.shape
    delta.tofile('out-%d/%02d-%02d-%02d-delta.raw' % (Nrep, i, j, k))
  
  with sharedmem.Pool() as pool:
    pool.map(work, range(Nrep * Nrep))
  
