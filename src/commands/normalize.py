import numpy
from cosmology import interp1d
from args import pixeldtype2

def main(A):
  """ normalize to observed mean flux """
  zcenter, afactor = numpy.loadtxt(A.datadir + '/afactor.txt', unpack=True)
  AZ = interp1d(zcenter, afactor, fill_value=1.0, kind=4)

  all = numpy.fromfile(A.datadir + '/pass3.raw', pixeldtype2)
  if A.redshift_distortion:
    Z = all['Z0'] + all['Zshift']
  else:
    Z = all['Z0']
  all['F'] = all['F'] ** AZ(Z)
  all.tofile(A.datadir + '/pass4.raw')
