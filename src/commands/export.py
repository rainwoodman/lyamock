import numpy
from args import pixeldtype1, pixeldtype2
from args import writefill, bitmapdtype
import sharedmem

def main(A):
    """ check """
    bitmap = numpy.fromfile('bitmap.raw', dtype=bitmapdtype).reshape(-1,
            A.Npixel)
    good = ~numpy.isnan(bitmap['F']).any(axis=-1)
    good = bitmap[good]
    verygood = good['lambda'] < 1185
    verygood = good[verygood]
    pos = verygood['pos'] - A.BoxSize * 0.5
    R = numpy.einsum('ij,ij->i', pos, pos) ** 0.5
    dec = numpy.arcsin(pos[:, 2] / R) / numpy.pi * 180
    ra = numpy.arctan2(pos[:, 1], pos[:, 0]) / numpy.pi * 180
    z = verygood['Z']
    numpy.savetxt(A.datadir + '/pixels.txt', numpy.array([ra, dec, z, verygood['F']]).T, fmt='%.8g')
