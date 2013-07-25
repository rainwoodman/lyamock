import numpy
import fitsio
import os
from args import bitmapdtype

def main(A):
    """ export """
    bitmap = numpy.fromfile(A.datadir + '/bitmap.raw',
            dtype=bitmapdtype)

    object = bitmap['objectid'].copy()
    left = object.searchsorted(numpy.arange(len(A.sightlines)), side='left')
    right = object.searchsorted(numpy.arange(len(A.sightlines)), side='right')
    for i in range(len(A.sightlines)):
        tofits(A, i, bitmap, left, right)

def tofits(A, i, bitmap, left, right):
    line = A.sightlines[i]
    filename = A.prefix + \
    '/%(refPLATE)04d/mockSmoothed-%(refPLATE)04d-%(refMJD)05d-%(refFIBERID)04d.fits' \
        % line
    
#    if line['refPLATE'] != 3586 or line['refFIBER'] != 498: 
#        return

    spectra = bitmap[left[i]:right[i]]
    assert (spectra['objectid'] == i).all()
    mask = ~numpy.isnan(spectra['flux'])
    spectra = spectra[mask]

    if len(spectra) == 0: return
    print filename, len(spectra)
    header = {}
    header['M_NGRID'] = len(spectra)
    header['NMOCKS'] = 1
    header['M_ZFID'] = 0
    header['M_Z'] = line['Z']
    header['M_DV'] = 0
    header['M_DEC'] = line['DEC']
    header['M_RA'] = line['RA']
    mock = numpy.empty(shape=len(spectra),
            dtype=[
                ('f', 'f4'),
                ('fdla', 'f4'),
                ('delta', 'f4'),
                ('lambda', 'f4'),
                ('z', 'f4')
                ])

    mock['f'] = spectra['flux']
    mock['delta'] = spectra['delta']
    mock['fdla'] = spectra['flux']
    mock['lambda'] = spectra['lambda']
    mock['z'] = spectra['Z']
    try:
        fits = fitsio.FITS(filename, 'rw', clobber=True)
    except Exception as e:
        os.makedirs(A.prefix + \
                '/%(refPLATE)04d' % line)
        fits = fitsio.FITS(filename, 'rw', clobber=True)

    fits.write(mock)
    h = fits[0]
    for key in header:
        h.write_key(key, header[key]) 

    fits.close()

def txt(A):
    bitmap = bitmap.reshape(-1, A.Npixel)

    good = ~numpy.isnan(bitmap['flux']).any(axis=-1)
    good = bitmap[good]
    verygood = good['lambda'] < 1185
    verygood = good[verygood]
    print len(good)
    print len(verygood)
    pos = verygood['pos'] - A.BoxSize * 0.5
    R = numpy.einsum('ij,ij->i', pos, pos) ** 0.5
    dec = numpy.arcsin(pos[:, 2] / R) / numpy.pi * 180
    ra = numpy.arctan2(pos[:, 1], pos[:, 0]) / numpy.pi * 180
    z = verygood['Z']
    numpy.savetxt(A.datadir + '/spectra.txt', numpy.array([ra, dec, z,
        verygood['flux'], verygood['var'], verygood['w']]).T, fmt='%g')
