import numpy
import sharedmem
from scipy.optimize import leastsq
from args import pixeldtype2

def main(A):
  """find the normalization factor matching the mean flux"""
  zbins = numpy.linspace(A.Zmin, A.Zmax, 100)
  zcenter = 0.5 * (zbins[1:] + zbins[:-1])
  afactor = numpy.ones_like(zcenter)
  meanflux_expected = A.FPGAmeanflux(1 / (1 + zcenter))
  print 'zmax = ', A.Zmax

  all = numpy.array([], dtype=pixeldtype2)
  for i, j, k in A.yieldwork():
      try:
        fill = numpy.memmap(A.basename(i, j, k, 'pass2') +
              '.raw', mode='r', dtype=pixeldtype2)
      except IOError:
        continue
      all = numpy.append(all, fill)
  all.tofile(A.datadir + '/pass3.raw')
  Z = all['Z0'].copy()
  if A.redshift_distortion == True:
      Z += all['Zshift']
  Fall = all['F'].copy()
  dig = numpy.digitize(Z, zbins)
  ind = dig.argsort()
  dig = dig[ind]
  Fall = Fall[ind]
  del all
  for i in range(len(zcenter)):
    left = dig.searchsorted(i + 1, side='left')
    right = dig.searchsorted(i + 1, side='right')
    F = Fall[left:right]
    if F.size > 0:
      with sharedmem.Pool(use_threads=True) as pool:
        def cost(az):
          def work(F):
            return (F[F!=0] ** az[0]).sum(dtype='f8')
          Fsum = numpy.sum(pool.starmap(work,
              pool.zipsplit((F,))))
          Fmean = Fsum / F.size
          dist = (Fmean - meanflux_expected[i])
          return dist
        if i > 1:
          p0 = afactor[i - 1]
        else:
          p0 = 1.0
        p = leastsq(cost, p0, full_output=False)
        afactor[i] = p[0]
    else:
      afactor[i] = 1.0
    print i, afactor[i], F.mean(), F.min(), F.max(), F.size
  numpy.savetxt(A.datadir + '/afactor.txt', zip(zcenter, afactor),
          fmt='%g')
