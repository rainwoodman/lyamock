import numpy
import os.path
import StringIO
import ConfigParser
import argparse
from scipy.interpolate import interp1d
from cosmology import Cosmology

pixeldtype = numpy.dtype(
    [('delta', 'f4'), ('losdisp', 'f4'), ('row', 'i4'), ('ind', 'i4'), ('Z', 'f4')])

pixeldtype2 = numpy.dtype(
    [('F', 'f4'), ('Zshift', 'f4'), ('row', 'i4'), ('ind', 'i4'), ('Z0', 'f4')])

bitmapdtype = numpy.dtype([('Z', 'f4'), ('F', 'f4'), ('R', 'f4'), ('pos', ('f4',
    3))])
class Config(argparse.Namespace):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("paramfile", 
               help="the paramfile")
    parser.add_argument("mode", choices=['firstpass',
       'secondpass', 
       'thirdpass',
       'fourthpass',
       'fifthpass',
       ])
    parser.add_argument("--row",
               help="which row to do", type=int,
               default=None)
    parser.add_argument("--serial",
               help="serial", action='store_true', default=False)
    parser.add_argument("--no-redshift-distortion", dest='redshift_distortion', 
               help="without redshift distortion", action='store_false',
               default=True)

    def export(self, dict, names):
        for name in names:
            setattr(self, name, dict[name])
    def basename(self, i, j, k, post):
        return '%s/%02d-%02d-%02d-delta-%s' % (self.datadir, i, j, k, post)
    def FPGAmeanflux(self, a):
        return numpy.exp(-10**(self.FitB + self.FitA * numpy.log10(a)))

    def __init__(self, argv):
        Config.parser.parse_args(argv, self)

        config = ConfigParser.ConfigParser()
        str = file(self.paramfile).read().replace(';', ',').replace('#', ';')
        config.readfp(StringIO.StringIO(str))

        Seed = config.getint("IC", "Seed")
        BoxSize = config.getfloat("IC", "BoxSize")
        Nmesh = config.getint("IC", "Nmesh")
        NmeshEff = config.getint("IC", "NmeshEff")
        NLyaBox = config.getint("IC", "NLyaBox")
        Npixel = config.getint("IC", "Npixel")
        Geometry = config.get("IC", "Geometry")
        Redshift = config.getfloat("IC", "Redshift")

        assert Geometry in ['Sphere', 'Test']

        assert NmeshEff % Nmesh  == 0
        Nrep = NmeshEff // Nmesh

        self.export(locals(), [
            'Seed', 'BoxSize', 'Redshift', 'Nmesh',
            'NmeshEff', 'NLyaBox', 'Nrep', 'Geometry', 'Npixel'])

        Sigma8 = config.getfloat("Cosmology", "Sigma8")
        OmegaM = config.getfloat("Cosmology", "OmegaM")
        OmegaB = config.getfloat("Cosmology", "OmegaB")
        OmegaL = config.getfloat("Cosmology", "OmegaL")
        h = config.getfloat("Cosmology", "h")
        G = 43007.1
        C = 299792.458
        H = 0.1
        DH = C / H
        cosmology = Cosmology(M=OmegaM, 
            L=OmegaL, B=OmegaB, h=h, sigma8=Sigma8)

        self.export(locals(), [
            'Sigma8', 'OmegaM', 'OmegaB', 'OmegaL',
            'h', 'G', 'C', 'H', 'DH', 'cosmology'])

        beta = config.getfloat("FPGA", "beta")
        JeansScale = config.getfloat("FPGA", "JeansScale")
        FitA = config.getfloat("FPGA", "FitA")
        FitB = config.getfloat("FPGA", "FitB")
        NmeshLya = int(BoxSize / NmeshEff / JeansScale * 0.25) * 4
        print 'High res Lya fluctuation is NmeshLya = ', NmeshLya
        print 'JeansScale is', JeansScale

        self.export(locals(), [
            'beta', 'JeansScale', 'FitA', 'FitB',
            'NmeshLya'] )


        numpy.random.seed(Seed)

        datadir = config.get("IO", "datadir")
        sightlinedtype = [('x1', ('f4', 3)),
                       ('x2', ('f4', 3)),
                       ('Zmax', 'f4'),
                       ('Zmin', 'f4')]
        if Geometry == 'Sphere':
            linesfile = config.get("IO", 'sightlines')
            raw = numpy.loadtxt(linesfile, usecols=(4, 5, 6, 0), 
                      dtype=[('RA', 'f4'), 
                             ('DEC', 'f4'), 
                             ('Z_VI', 'f4'), 
                             ('THINGID', 'i8')]).view(numpy.recarray)
            sightlines = numpy.empty(raw.size, 
                    dtype=sightlinedtype).view(numpy.recarray)
            sightlines.Zmax = raw.Z_VI
            sightlines.Zmin = (raw.Z_VI + 1) * 1026. / 1216 - 1
            sightlines.Zmin.clip(min=Redshift, out=sightlines.Zmin)
            raw.RA *= numpy.pi / 180
            raw.DEC *= numpy.pi / 180
            sightlines.x1[:, 0] = numpy.cos(raw.RA) * numpy.sin(raw.DEC)
            sightlines.x1[:, 1] = numpy.sin(raw.RA) * numpy.sin(raw.DEC)
            sightlines.x1[:, 2] = numpy.cos(raw.DEC)
            sightlines.x2[...] = sightlines.x1
            Dcmin = cosmology.Dc(1 / (sightlines.Zmin + 1))[:, None]
            Dcmax = cosmology.Dc(1 / (sightlines.Zmax + 1))[:, None]

            sightlines.x1 *= Dcmin
            sightlines.x2 *= Dcmax
            sightlines.x1 *= DH
            sightlines.x2 *= DH
             
            sightlines.x1 += BoxSize * 0.5
            sightlines.x2 += BoxSize * 0.5
        elif Geometry == 'Test':
            sightlines = numpy.empty(Npixel ** 2, 
                    dtype=sightlinedtype).view(numpy.recarray)
            Zmin = Redshift
            Dcmin = cosmology.Dc(1 / (Zmin + 1))
            Zmax = 1 / cosmology.aback(Dcmin + BoxSize / DH) - 1
            sightlines.Zmax[...] = Zmax
            sightlines.Zmin[...] = Zmin
            x, y = \
                    numpy.indices((Npixel, Npixel)).reshape(2, -1) \
                    * (BoxSize / Npixel)
            sightlines.x1[:, 0] = x
            sightlines.x1[:, 1] = y
            sightlines.x1[:, 2] = 0
            sightlines.x2[...] = sightlines.x1 
            sightlines.x2[:, 2] = BoxSize

        assert (sightlines.x1 >= 0).all()
        assert (sightlines.x1 <= BoxSize).all()
        assert (sightlines.x2 >= 0).all()
        assert (sightlines.x2 <= BoxSize).all()

        Zmax = sightlines.Zmax.max()
        Zmin = sightlines.Zmin.min()
        self.export(locals(), [
            'datadir', 'sightlines', 'Zmax', 'Zmin'])
        print 'max Z is', Zmax, 'min Z is', Zmin

        try:
            power = config.get("Cosmology", "PowerSpectrum")
        except ConfigParser.NoOptionError:
            power = datadir + '/power.txt'
        try:
            k, p = numpy.loadtxt(power, unpack=True)
            print 'using power from file ', power
        except IOError:
            print 'using power from pycamb, saving to ', power
            Pk = cosmology.Pk
            k = Pk.x / DH
            p = Pk.y * DH ** 3 * (2 * numpy.pi) ** -3
            numpy.savetxt(power, zip(k, p))

        p[numpy.isnan(p)] = 0
        power = interp1d(k, p, kind='linear', copy=True, 
                         bounds_error=False, fill_value=0)
        self.export(locals(), ['power'])

    def yieldwork(self):
        if self.row is not None:
            for jk in range(self.Nrep * self.Nrep):
                j, k = numpy.unravel_index(jk, (self.Nrep, ) * 2)
                yield self.row, j, k
        else:
            for ijk in range(self.Nrep * self.Nrep * self.Nrep):
                i, j, k = numpy.unravel_index(ijk, (self.Nrep, ) * 3)
                yield i, j, k
    
def parseargs(argv=None):
  return Config(argv)


def loadpower(args):
    return func

def loadpixel(args, i, j, k):
  if args.pixeldir is not None:
    try:
      return numpy.fromfile(args.pixeldir + "%02d-%02d-%02d-pixels.raw" % (i, j, k), dtype=pixeldtype)
    except IOError:
      return numpy.array([], dtype=pixeldtype)
  else:
    Offset = numpy.array(numpy.unravel_index(numpy.arange(args.Nrep ** 3),
             (args.Nrep, ) * 3)).T.reshape(args.Nrep, args.Nrep, args.Nrep, 3) \
               * args.BoxSize / args.Nrep
    Npix = args.Npix / args.Nrep
    intpos = numpy.array(numpy.unravel_index(numpy.arange(Npix ** 3),
             (Npix, ) * 3)).T
    rt = numpy.empty(shape=len(intpos), dtype=pixeldtype)
    rt['pos'] = Offset[i, j, k] + intpos * (args.BoxSize / args.Npix)
    DH = args.C / args.H
    dist = rt['pos'][:, 2] + DH * args.cosmology.Dc(1 / (1. + args.Redshift))
    rt['Z'] = 1 / args.cosmology.aback(dist / DH) - 1
    rt['row'] = (i * Npix+ intpos[:, 0]) * args.Npix + j * Npix + intpos[:, 1]
    rt['ind'] = intpos[:, 2] + k * Npix
    return rt

def writefill(fill, basename, stat=None):
    fill.tofile(basename + '.raw')
    if stat is not None:
      K2 = (fill['delta'] ** 2).sum(dtype='f8')
      K = fill['delta'].sum(dtype='f8')
      N = fill['delta'].size
      file(basename + '.info-file', mode='w').write(
"""
K2 = %(K2)g
K = %(K)g
N = %(N)g
""" % locals())
    print 'wrote', basename
