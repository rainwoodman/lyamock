import numpy
from splat import splat
import sharedmem
from args import pixeldtype, bitmapdtype

def main(A):
    """rebin to observed sight lines, this going to take hack lot of memory"""
    with A.F('bitmap', mode='w'):
        pass

    offset = sharedmem.empty(len(A.sightlines), dtype='i8')
    length = sharedmem.empty(len(A.sightlines), dtype='i8')
  
    objectid = A.P('objectid')
    ind = sharedmem.argsort(objectid)
    sorted = objectid[ind]
    start = sharedmem.searchsorted(sorted, 
            numpy.arange(len(offset)), side='left')
    end = sharedmem.searchsorted(sorted, 
            numpy.arange(len(offset)), side='right')
  
    del sorted, objectid
    data = lambda : None
    data.Zreal = A.P('Zreal')
    data.delta = A.P('delta')
    if not A.skipred:
        data.disp = numpy.array([
            A.P('dispx'),
            A.P('dispy'),
            A.P('dispz')
            ]).T

    if not A.usepass1:
        data.flux = A.P('flux')
 
    disp_to_vel = A.cosmology.disp_to_vel
    Dc = A.cosmology.Dc
    redshift_dist = A.cosmology.redshift_dist

    with sharedmem.Pool() as pool:
      def work(i):
        Zmax, Zmin, Zqso = (A.sightlines.Zmax[i], 
                             A.sightlines.Zmin[i],
                             A.sightlines.Z[i])
        x1, x2 = A.sightlines.x1[i], A.sightlines.x2[i]
        dir = A.sightlines.dir[i] 

        if Zmin >= Zmax: 
            length[i] = 0
            return

        # avoid marking pages dirty
        subset = ind[start[i]:end[i]].copy()
        subset.sort()
        with pool.critical:
            Zreal = data.Zreal[subset]
            delta = data.delta[subset]

            if not A.usepass1:
                flux = data.flux[subset]
        if not A.skipred:
            disp = numpy.einsum('ij, j -> i', data.disp[subset], dir)
            a = 1 / (Zreal + 1)
            vel = disp_to_vel(disp / A.DH, a)
            Zred = 1 / redshift_dist(vel, a) - 1

        if A.Geometry == 'Test':
            # in test mode, the bins are uniform in comoving distance
            Dcmax = Dc(1 / (1 + Zmax))
            Dcmin = Dc(1 / (1 + Zmin))
            bins = 1 / Dc.inv(numpy.linspace(Dcmin, Dcmax, A.Npixel + 1,
                endpoint=True)) - 1
            R1 = Dcmin * A.DH
            R2 = Dcmax * A.DH
        else:
            R1 = ((x1 - A.BoxSize * 0.5)** 2).sum() ** 0.5
            R2 = ((x2 - A.BoxSize * 0.5)** 2).sum() ** 0.5
          #  llMin, llMax = numpy.log10([1026., 1216.])
          #  lambdagrid = 10 ** numpy.linspace(llMin, llMax, A.Npixel + 1, endpoint=True)
            Npix = int((1216 - 1026) * (1 + Zqso))
            lambdagrid = numpy.linspace(1026., 1216., Npix + 1, endpoint=True)
            bins = (Zqso + 1.) * lambdagrid / 1216. - 1

        spectra = numpy.empty(len(bins) - 1, dtype=bitmapdtype)

        spectra['Z'] = (bins[1:] + bins[:-1]) * 0.5
        spectra['lambda'] = (spectra['Z'] + 1) / (1 + Zqso) * 1216.
  
        R = A.cosmology.Dc(1 / (1 + spectra['Z'])) * A.DH
        for d in numpy.arange(3):
          spectra['pos'][:, d] = numpy.interp(R, 
                  [R1, R2], 
                  [x1[d], x2[d]])
        spectra['objectid'] = i

        # do the real
        Wreal = splat(Zreal, 1, bins)
        assert numpy.allclose(Wreal.sum(), len(Zreal))
        Wreal = Wreal[1:-1]

        if not A.usepass1:
            spectra['flux'] = splat(Zreal, flux, bins)[1:-1] / Wreal
        spectra['delta'] = splat(Zreal, delta, bins)[1:-1] / Wreal
        spectra['wreal'] = Wreal

        # do the red
        if not A.skipred:
            Wred = splat(Zred, 1, bins)[1:-1]
            if not A.usepass1:
                spectra['fluxred'] = splat(Zred, flux, bins)[1:-1] / Wred
            spectra['deltared'] = splat(Zred, delta, bins)[1:-1] / Wred
            spectra['wred'] = Wred

        # these points have no samples and we fake them by directly using the
        # mean flux.
        # lack = W[1:-1] == 0
        # bitmap['F'][i][lack] = A.FPGAmeanflux(1 / (1 + bitmap['Z'][i][lack]))
        # no we actually leave them NaN
        spectra = spectra[ \
                (~numpy.isnan(spectra['flux']) & \
                 ~numpy.isnan(spectra['fluxred'])) \
                 ]
        length[i] = len(spectra)
        with pool.ordered:
            with A.F('bitmap', mode='a') as f:
                spectra.tofile(f)
                f.flush()
      pool.map(work, numpy.arange(len(length)))

    offset[1:] = length.cumsum()[:-1]
    offset[0] = 0
    length.tofile(A.datadir + '/bitmap-length.raw')
    offset.tofile(A.datadir + '/bitmap-offset.raw')

