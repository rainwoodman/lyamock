import numpy
from splat import splat
import sharedmem
from args import pixeldtype2, bitmapdtype

def main(A):
  """rebin to observed sight lines, this going to take hack lot of memory"""
  bitmap = sharedmem.empty((A.sightlines.size, A.Npixel), 
          bitmapdtype)
  all = numpy.fromfile(A.datadir + '/pass4.raw', pixeldtype2)
  arg = all['row'].argsort()
  all = all[arg]

  start = all['row'].searchsorted(numpy.arange(len(bitmap)), side='left')
  end = all['row'].searchsorted(numpy.arange(len(bitmap)), side='right')
  Zall = all['Z0']

  if A.redshift_distortion:
      Zall = Zall + all['Zshift']

  def work(range):
    for i in range:
      Z = Zall[start[i]:end[i]]
      F = all['F'][start[i]:end[i]]
      if A.Geometry == 'Test':
          # in test mode, the bins are uniform in comoving distance
          Zmax, Zmin = A.sightlines.Zmax[i], A.sightlines.Zmin[i]
          Dcmax = A.cosmology.Dc(1 / (1 + Zmax))
          Dcmin = A.cosmology.Dc(1 / (1 + Zmin))
          bins = 1 / A.cosmology.aback(numpy.linspace(Dcmin, Dcmax, A.Npixel + 1,
              endpoint=True)) - 1
      else:
          Zmax, Zmin = A.sightlines.Zmax[i], A.sightlines.Zmin[i]
          llMin, llMax = numpy.log10([(Zmin + 1) /  (Zmax + 1) * 1216, 1216])
          lambdagrid = 10 ** numpy.linspace(llMin, llMax, A.Npixel + 1, endpoint=True)
          bins = (Zmax + 1.) * lambdagrid / 1216. - 1
      bitmap['Z'][i] = (bins[1:] + bins[:-1]) * 0.5
      R = A.cosmology.Dc(1 / (1 + bitmap['Z'][i])) * A.DH
      bitmap['R'][i] = R
      W = splat(Z, 1, bins)
      x1, x2 = A.sightlines.x1[i], A.sightlines.x2[i]
      R1 = ((x1 - A.BoxSize * 0.5)** 2).sum()
      R2 = ((x2 - A.BoxSize * 0.5)** 2).sum()
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
      pool.map(work, numpy.array_split(numpy.arange(len(bitmap)), len(bitmap) / 512))
  bitmap.tofile(A.datadir + '/bitmap.raw')
