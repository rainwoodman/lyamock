import numpy
from args import bitmapdtype

def main(A):
    """ export """
    bitmap = numpy.fromfile(A.datadir + '/bitmap.raw', dtype=bitmapdtype)
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

