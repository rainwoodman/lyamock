raise deprecated
import numpy
from cosmology import interp1d
from args import pixeldtype2

def main(A):
  """ normalize to observed mean F """
  zcenter, afactor = numpy.loadtxt(A.datadir + '/afactor.txt', unpack=True)
  AZ = interp1d(zcenter, afactor, fill_value=1.0, kind=4)

  all = numpy.fromfile(A.datadir + '/pass3.raw', dtype=pixeldtype2)
  Z = A.Z(all)
  all['F'] = all['F'] ** AZ(Z)
  all.tofile(A.datadir + '/pass4.raw')
