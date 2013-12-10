import numpy
import sharedmem
from args import pixeldtype, bitmapdtype
from splat import splat
from convolve import IndexBySightline
def main(A):
    """rebin to observed sight lines, this going to take hack lot of memory"""
    with A.F('bitmap', mode='w'):
        pass

    delta = A.P('delta')
    taured = A.P('taured')
    taureal = A.P('taureal')
    dc = A.P('dc')

    Afactor = numpy.loadtxt(A.datadir + '/afactors.txt').T
#    Afactor[1][:] = 0.001
    index = IndexBySightline(A)

    def makespectra(i):
        Zqso = A.sightlines.Z[i]
        Zmax = (Zqso + 1) * 1216. / 1216 - 1
        Zmin = (Zqso + 1) * 1026. / 1216 - 1

        Rmin = A.cosmology.Dc(1 / (Zmin + 1)) * A.DH
        Rmax = A.cosmology.Dc(1 / (Zmax + 1)) * A.DH
        ind = index[i]

        dir = A.sightlines.dir[i] 

        Rbins = numpy.arange(Rmin, Rmax + A.PixelScale, A.PixelScale) 
        Rbincenter = (Rbins[1:] + Rbins[:-1]) * 0.5
        spectra = numpy.empty(len(Rbincenter), dtype=bitmapdtype)
        a = A.cosmology.Dc.inv(Rbincenter / A.DH)
        spectra['Z'] = 1 / a - 1
        spectra['lambda'] = (spectra['Z'] + 1)/ (1 + Zqso) * 1216.
        spectra['objectid'] = i

        # do the real
        R = dc[ind] * A.DH
#        print 'dc, A', dc[ind][0], numpy.interp(dc[ind][0], Afactor[0], Afactor[1])
        F = numpy.exp(-taured[ind] * 
                numpy.interp(dc[ind], Afactor[0], Afactor[1]))
        Freal = numpy.exp(-taureal[ind] * 
                numpy.interp(dc[ind], Afactor[0], Afactor[1]))
#        print spectra['Z'].mean(), A.FPGAmeanF(1 / (spectra['Z'].mean() + 1))
        W = splat(R, 1, Rbins)[1:-1]
        spectra['F'] = splat(R, F, Rbins)[1:-1] / W
        spectra['Freal'] = splat(R, Freal, Rbins)[1:-1] / W
#        print F.mean(), spectra['F'].mean()
        spectra['delta'] = splat(R, 1 + delta[ind], Rbins)[1:-1] \
                * (A.JeansScale / A.PixelScale) - 1
        spectra['pos'] = (Rbincenter - A.sightlines.Rmin[i])[:, None] /\
                (A.sightlines.Rmax[i] - A.sightlines.Rmin[i]) * \
                (A.sightlines.x2[i] - A.sightlines.x1[i])[None, :] \
                + A.sightlines.x1[i][None, :]
        if i == 0:
            numpy.savez('spectra.npz', 
                    spectra=spectra, R=Rbincenter)

        # these points have no samples and we fake them by directly using the
        # mean F.
        # lack = W[1:-1] == 0
        # bitmap['F'][i][lack] = A.FPGAmeanF(1 / (1 + bitmap['Z'][i][lack]))
        # no we actually leave them NaN
        return spectra
    chunksize = 1024
    with sharedmem.Pool() as pool:
        def work(start):
            lines = []
            for i in range(*slice(start, start +
                chunksize).indices(len(A.sightlines))):
                spectra = makespectra(i)
                lines.append(spectra)

            with pool.ordered:
                print 'writing bunch', i
                with A.F('bitmap', mode='a') as f:
                    for spectra in lines:
                        spectra.tofile(f)
                    f.flush()

        pool.map(work, range(0, len(A.sightlines), chunksize))

