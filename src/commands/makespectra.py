import numpy
from splat import splat
import sharedmem
from args import pixeldtype, bitmapdtype

def main(A):
    """rebin to observed sight lines, this going to take hack lot of memory"""
    bitmap = numpy.memmap(
            A.datadir + '/bitmap.raw',
            shape=(A.sightlines.size, A.Npixel), 
            dtype=bitmapdtype,
            mode='w+'
            )
  
    objectid = A.P('objectid')
    ind = sharedmem.argsort(objectid)
    sorted = objectid[ind]
    start = sharedmem.searchsorted(sorted, 
            numpy.arange(len(bitmap)), side='left')
    end = sharedmem.searchsorted(sorted, 
            numpy.arange(len(bitmap)), side='right')
  
    del sorted
    if A.RedshiftDistortion:
        Z = A.P('Zred')
    else:
        Z = A.P('Zreal')

    delta = A.P('delta')
    if not A.usepass1:
        flux = A.P('flux')

    def work(i):
        spectra = bitmap[i]
        Zmax, Zmin, Zqso = (A.sightlines.Zmax[i], 
                             A.sightlines.Zmin[i],
                             A.sightlines.Z[i])
        x1, x2 = A.sightlines.x1[i], A.sightlines.x2[i]
    
        if Zmin >= Zmax: 
            spectra['flux'] = numpy.nan
            spectra['delta'] = numpy.nan
            spectra['Z'] = numpy.nan
            spectra['lambda'] = numpy.nan
            spectra['pos'] = numpy.nan
            spectra['objectid'] = i
            spectra['w'] = numpy.nan
            spectra['var'] = numpy.nan
            return

        # avoid marking pages dirty
        subset = ind[start[i]:end[i]].copy()
        subset.sort()
        z = Z[subset]
        if not A.usepass1:
            F = flux[subset]
        D = delta[subset]
  
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
        spectra['Z'] = (bins[1:] + bins[:-1]) * 0.5
        spectra['lambda'] = (spectra['Z'] + 1) / (1 + Zqso) * 1216.
  
        R = A.cosmology.Dc(1 / (1 + spectra['Z'])) * A.DH
        spectra['objectid'] = i

        W = splat(z, 1, bins)
        for d in numpy.arange(3):
          spectra['pos'][:, d] = numpy.interp(R, 
                  [R1, R2], 
                  [x1[d], x2[d]])
        assert numpy.allclose(W.sum(), len(z))
        if not A.usepass1:
            spectra['flux'] = splat(z, F, bins)[1:-1] / W[1:-1]
        spectra['delta'] = splat(z, D, bins)[1:-1] / W[1:-1]
        spectra['w'] = W[1:-1]
        if not A.usepass1:
            flux2 = splat(z, F ** 2, bins)[1:-1] / W[1:-1]
            spectra['var'] = flux2 - spectra['flux'] ** 2
        # these points have no samples and we fake them by directly using the
        # mean flux.
        # lack = W[1:-1] == 0
        # bitmap['F'][i][lack] = A.FPGAmeanflux(1 / (1 + bitmap['Z'][i][lack]))
        # no we actually leave them NaN
    with sharedmem.Pool() as pool:
        pool.map(work, numpy.arange(len(bitmap)))
    #    pool.map(work, [13112])
    bitmap.flush()
