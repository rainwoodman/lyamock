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
    data.Zreal = A.P('Zreal', memmap='r')
    data.delta = A.P('delta', memmap='r')
    if not A.skipred:
        if A.Geometry == 'Test':
            data.disp = (
                None,
                None,
                A.P('dispz', memmap='r') )
        else:
            data.disp = (
                A.P('dispx', memmap='r'),
                A.P('dispy', memmap='r'),
                A.P('dispz', memmap='r') )

    if not A.usepass1:
        data.flux = A.P('flux', memmap='r')
 
    disp_to_vel = A.cosmology.disp_to_vel
    Dc = A.cosmology.Dc
    redshift_dist = A.cosmology.redshift_dist

    def makespectra(i):
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
            if A.Geometry == 'Test':
                disp = data.disp[2][subset]
            else:
                disp = data.disp[0][subset] * dir[0] \
                     + data.disp[1][subset] * dir[1] \
                     + data.disp[2][subset] * dir[2]

            a = 1 / (Zreal + 1)
            vel = disp_to_vel(disp / A.DH, a)
            Zred = 1 / redshift_dist(vel, a) - 1

            if A.Geometry == 'Test':
                # not exactly right but will move pixels
                # into the box
                disp[Zred > Zmax] -= A.BoxSize
                disp[Zred < Zmin] += A.BoxSize
                vel = disp_to_vel(disp / A.DH, a)
                Zred = 1 / redshift_dist(vel, a) - 1

        # bins are uniform in comoving distance
        Dcmax = Dc(1 / (1 + Zmax))
        Dcmin = Dc(1 / (1 + Zmin))
        if A.Geometry == 'Test':
            bins = numpy.linspace(Dcmin, Dcmax, A.Npixel + 1, endpoint=True)
        else:
            # about 12 nm per bin (1216 - 1026 = 190.)
            bins = numpy.linspace(Dcmin, Dcmax, int(150 * (1 + Zqso)), endpoint=True)
        R1 = Dcmin * A.DH
        R2 = Dcmax * A.DH

        spectra = numpy.empty(len(bins) - 1, dtype=bitmapdtype)

        spectra['Z'] = 1 / A.cosmology.Dc.inv((bins[1:] + bins[:-1]) * 0.5) - 1
        spectra['lambda'] = (spectra['Z'] + 1) / (1 + Zqso) * 1216.
  
        R = A.cosmology.Dc(1 / (1 + spectra['Z'])) * A.DH
        for d in numpy.arange(3):
          spectra['pos'][:, d] = numpy.interp(R, 
                  [R1, R2], 
                  [x1[d], x2[d]])
        spectra['objectid'] = i

        pixwidth = (Dcmax - Dcmin) / len(Zreal)
        binwidth = numpy.diff(bins)
        Xreal = A.cosmology.Dc(1 / (1 + Zreal))
        Xred  = A.cosmology.Dc(1 / (1 + Zred))
        # do the real
        Wreal = splat(Xreal, 1, bins)
        assert numpy.allclose(Wreal.sum(), len(Zreal))
        Wreal = Wreal[1:-1]

        if not A.usepass1:
            spectra['flux'] = splat(Xreal, flux, bins)[1:-1] / Wreal
        spectra['delta'] = splat(Xreal, 1 + delta, bins)[1:-1] \
                * (pixwidth / binwidth) - 1
        spectra['wreal'] = Wreal

        if i == 0:
            numpy.savez('spectra.npz', spectra=spectra, Zreal=Zreal,
                    delta=delta, bins=bins, binwdith=binwidth)
        # do the red
        if not A.skipred:
            Wred = splat(Xred, 1, bins)[1:-1]
            if not A.usepass1:
                spectra['fluxred'] = splat(Xred, flux, bins)[1:-1] / Wred
            spectra['deltared'] = splat(Xred, 1 + delta, bins)[1:-1] \
                * (pixwidth / binwidth) - 1
            spectra['wred'] = Wred

        # these points have no samples and we fake them by directly using the
        # mean flux.
        # lack = W[1:-1] == 0
        # bitmap['F'][i][lack] = A.FPGAmeanflux(1 / (1 + bitmap['Z'][i][lack]))
        # no we actually leave them NaN
        if A.Geometry != 'Test':
            spectra = spectra[ \
                    (~numpy.isnan(spectra['flux']) & \
                     ~numpy.isnan(spectra['fluxred']) & \
                     ~numpy.isnan(spectra['delta']) & \
                     ~numpy.isnan(spectra['deltared'])) \
                     ]
        return spectra
    chunksize = 1024
    with sharedmem.Pool() as pool:
        def work(start):
            lines = []
            for i in range(*slice(start, start +
                chunksize).indices(len(A.sightlines))):
                spectra = makespectra(i)
                length[i] = len(spectra)
                lines.append(spectra)

            with pool.ordered:
                with A.F('bitmap', mode='a') as f:
                    for spectra in lines:
                        spectra.tofile(f)
                    f.flush()

        pool.map(work, range(0, len(A.sightlines), chunksize))

    offset[1:] = length.cumsum()[:-1]
    offset[0] = 0
    length.tofile(A.datadir + '/bitmap-length.raw')
    offset.tofile(A.datadir + '/bitmap-offset.raw')

