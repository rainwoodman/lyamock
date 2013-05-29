import numpy
from splat import splat
import sharedmem
from args import pixeldtype2, bitmapdtype

def main(A):
  """rebin to observed sight lines, this going to take hack lot of memory"""
  bitmap = sharedmem.empty((A.sightlines.size, A.Npixel), 
          bitmapdtype)
  if A.usepass1:
    all = numpy.fromfile(A.datadir + '/pass1.raw', pixeldtype2)
  else:
    all = numpy.fromfile(A.datadir + '/pass4.raw', pixeldtype2)

  arg = sharedmem.argsort(all['row'])
  all = all[arg]

  start = sharedmem.searchsorted(all['row'], 
          numpy.arange(len(bitmap)), side='left')
  end = sharedmem.searchsorted(all['row'], 
          numpy.arange(len(bitmap)), side='right')
  Zall = all['Z0']

  Zall = A.Z(all)

  def work(range):
    for i in range:
      Z = Zall[start[i]:end[i]]
      F = all['F'][start[i]:end[i]]
      Zmax, Zmin, Zqso = A.sightlines.Zmax[i],  \
                         A.sightlines.Zmin[i],  \
                         A.sightlines.Z[i] 

      if Zmin >= Zmax: 
          bitmap['F'][i] = numpy.nan
          bitmap['Z'][i] = numpy.nan
          bitmap['lambda'][i] = numpy.nan
          bitmap['pos'][i] = numpy.nan
          bitmap['row'][i] = i
          continue

      x1, x2 = A.sightlines.x1[i], A.sightlines.x2[i]
      if A.Geometry == 'Test':
          # in test mode, the bins are uniform in comoving distance
          Dcmax = A.cosmology.Dc(1 / (1 + Zmax))
          Dcmin = A.cosmology.Dc(1 / (1 + Zmin))
          bins = 1 / A.cosmology.aback(numpy.linspace(Dcmin, Dcmax, A.Npixel + 1,
              endpoint=True)) - 1
          R1 = Dcmin * A.DH
          R2 = Dcmax * A.DH
      else:
          R1 = ((x1 - A.BoxSize * 0.5)** 2).sum() ** 0.5
          R2 = ((x2 - A.BoxSize * 0.5)** 2).sum() ** 0.5
          llMin, llMax = numpy.log10([1026., 1216.])
          lambdagrid = 10 ** numpy.linspace(llMin, llMax, A.Npixel + 1, endpoint=True)
          bins = (Zqso + 1.) * lambdagrid / 1216. - 1
      bitmap['Z'][i] = (bins[1:] + bins[:-1]) * 0.5
      bitmap['lambda'][i] = (bitmap['Z'][i] + 1) / (1 + Zqso) * 1216.

      R = A.cosmology.Dc(1 / (1 + bitmap['Z'][i])) * A.DH
      bitmap['row'][i] = i
      W = splat(Z, 1, bins)
      for d in numpy.arange(3):
        bitmap['pos'][i][:, d] = numpy.interp(R, [R1, R2], [x1[d],
          x2[d]])
      assert numpy.allclose(W.sum(), len(Z))
      bitmap['F'][i] = splat(Z, F, bins)[1:-1] / W[1:-1]
      # these points have no samples and we fake them by directly using the
      # mean flux.
      # lack = W[1:-1] == 0
      # bitmap['F'][i][lack] = A.FPGAmeanflux(1 / (1 + bitmap['Z'][i][lack]))
  with sharedmem.Pool() as pool:
      pool.map(work, numpy.array_split(numpy.arange(len(bitmap)), len(bitmap) /
          512 + 1))
  bitmap.tofile(A.datadir + '/bitmap.raw')
