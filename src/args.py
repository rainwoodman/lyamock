import numpy
import os.path
import StringIO
import ConfigParser
import argparse
from scipy.interpolate import interp1d
from cosmology import Cosmology, Lazy

pixeldtype0 = numpy.dtype(
    [('delta', 'f4'), ('losdisp', 'f4'), ('row', 'i4'), ('ind', 'i4'), ('Z', 'f4')])

pixeldtype1 = numpy.dtype(
    [('delta', 'f4'), ('Zshift', 'f4'), ('row', 'i4'), ('ind', 'i4'), ('Z0', 'f4')])
pixeldtype2 = numpy.dtype(
    [('F', 'f4'), ('Zshift', 'f4'), ('row', 'i4'), ('ind', 'i4'), ('Z0', 'f4')])

bitmapdtype = numpy.dtype([('Z', 'f4'), ('F', 'f4'), ('lambda', 'f4'), ('pos', ('f4',
    3)), ('row', 'i4')])

class Config(argparse.Namespace):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("paramfile", 
               help="the paramfile")
    parser.add_argument("mode", choices=['firstpass',
       'secondpass', 
       'thirdpass',
       'fourthpass',
       'fifthpass',
       'check',
       'export',
       'sightlines',
       ])
    parser.add_argument("--row",
               help="which row to do", type=int,
               default=None)
    parser.add_argument("--usepass1",
               help="use pass1 input in fifthpass, \
                     thus the bitmap will be density", 
               action='store_true', default=False)
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

    def Z(self, pixels):
        if self.redshift_distortion:
            return pixels['Z0'] + pixels['Zshift']
        else:
            return pixels['Z0']

    def FPGAmeanflux(self, a):
        return numpy.exp(-10**(self.FitB + self.FitA * numpy.log10(a)))

    def __init__(self, argv):
        Config.parser.parse_args(argv, self)

        config = ConfigParser.ConfigParser()
        str = file(self.paramfile).read().replace(';', ',').replace('#', ';')
        config.readfp(StringIO.StringIO(str))

        self.config = config

        Seed = config.getint("IC", "Seed")
        BoxSize = config.getfloat("IC", "BoxSize")
        NmeshCoarse = config.getint("IC", "NmeshCoarse")
        NmeshEff = config.getint("IC", "NmeshEff")
        Nrep = config.getint("IC", "Nrep")
        NLyaBox = config.getint("IC", "NLyaBox")
        Npixel = config.getint("IC", "Npixel")
        Geometry = config.get("IC", "Geometry")
        Redshift = config.getfloat("IC", "Redshift")

        Observer = numpy.array([BoxSize * 0.5] * 3, dtype='f8')
        assert Geometry in ['Sphere', 'Test']

        assert NmeshEff % Nrep == 0
        assert NmeshEff % NmeshCoarse == 0

        NmeshFine = NmeshEff / Nrep

        RNG = numpy.random.RandomState(Seed)
        SeedTable = RNG.randint(1<<21 - 1, size=(Nrep,) * 3)

        self.export(locals(), [
            'Seed', 'RNG', 'SeedTable', 'BoxSize', 'Redshift', 'NmeshFine', 'NmeshCoarse',
            'NmeshEff', 'NLyaBox', 'Nrep', 'Geometry', 'Npixel', 'Observer'])

        print 'NmeshFine is', NmeshFine 
        print 'Nrep is', Nrep, 'grid', BoxSize / Nrep
        print 'NmeshEff is', NmeshEff, 'grid', BoxSize / NmeshEff
        print 'NmeshCoarse is', NmeshCoarse, 'grid', BoxSize / NmeshCoarse

        Sigma8 = config.getfloat("Cosmology", "Sigma8")
        OmegaM = config.getfloat("Cosmology", "OmegaM")
        OmegaB = config.getfloat("Cosmology", "OmegaB")
        OmegaL = config.getfloat("Cosmology", "OmegaL")
        h = config.getfloat("Cosmology", "h")
        G = 43007.1
        C = 299792.458
        H0 = 0.1
        DH = C / H0
        cosmology = Cosmology(M=OmegaM, 
            L=OmegaL, B=OmegaB, h=h, sigma8=Sigma8)

        self.export(locals(), [
            'Sigma8', 'OmegaM', 'OmegaB', 'OmegaL',
            'h', 'G', 'C', 'H0', 'DH', 'cosmology'])

        QSOScale = config.getfloat("FPGA", "QSOscale")
        beta = config.getfloat("FPGA", "beta")
        JeansScale = config.getfloat("FPGA", "JeansScale")
        FitA = config.getfloat("FPGA", "FitA")
        FitB = config.getfloat("FPGA", "FitB")

        NmeshLya = int(BoxSize / NmeshEff / JeansScale * 0.25) * 4
        NmeshQSO = int(BoxSize / Nrep / QSOScale * 0.25) * 4
        if NmeshLya < 1: NmeshLya = 1
        if NmeshQSO < 1: NmeshQSO = 1

        NmeshLyaEff = NmeshLya * NmeshEff
        NmeshQSOEff = NmeshQSO * Nrep
        print 'Lya Eff NmeshLyaEff = ', NmeshLyaEff, \
                 BoxSize / NmeshLyaEff
        print 'JeansScale is', JeansScale
        print 'NmeshQSOEff is', NmeshQSOEff, 'grid', BoxSize / NmeshQSOEff

        self.export(locals(), [
            'beta', 'JeansScale', 'FitA', 'FitB',
            'NmeshLya', 'NmeshLyaEff', 'QSOScale', 'NmeshQSO', 'NmeshQSOEff'] )


        datadir = config.get("IO", "datadir")
        self.export(locals(), ['datadir'])

    @Lazy
    def QSOpercentile(self):
        """ returns a function evaluating 
            QSO peak percentile at given R
        """
        config = self.config
        try:
            QSOdensity = config.get("FPGA", "QSOdensityfile")

            zbins = numpy.load(QSOdensity)['bins']
            count = numpy.load(QSOdensity)['count']
            a = 1 / (zbins + 1.)
            R = self.cosmology.Dc(a) * self.DH
            V = numpy.diff(4. / 3. * numpy.pi * R ** 3)
            percentile = count / V / (self.NmeshQSO * self.Nrep / self.BoxSize) ** 3 
            return interp1d((R[1:] + R[:-1]) * .5, percentile, bounds_error=False,
                fill_value=0.0)
        except:
            QLF = config.get("FPGA", "QLFfile")

            z = numpy.load(QLF)['z']
            density = numpy.load(QLF)['density']
            a = 1 / (z + 1.)
            R = self.cosmology.Dc(a) * self.DH
            percentile = density / (self.NmeshQSO * self.Nrep / self.BoxSize) ** 3 
            print percentile.max()
            return interp1d(R, percentile, bounds_error=False, fill_value=0.0)

    @Lazy
    def power(self):
        config = self.config
        try:
            power = config.get("Cosmology", "PowerSpectrum")
        except ConfigParser.NoOptionError:
            power = self.datadir + '/power.txt'
        try:
            k, p = numpy.loadtxt(power, unpack=True)
            print 'using power from file ', power
        except IOError:
            print 'using power from pycamb, saving to ', power
            Pk = self.cosmology.Pk
            k = Pk.x / self.DH
            p = Pk.y * self.DH ** 3 * (2 * numpy.pi) ** -3
            numpy.savetxt(power, zip(k, p))

        p[numpy.isnan(p)] = 0
        power = interp1d(k, p, kind='linear', copy=True, 
                         bounds_error=False, fill_value=0)
        return power

    @Lazy
    def Zmin(self):
        return self.sightlines.Zmin.min()

    @Lazy
    def Zmax(self):
        return self.sightlines.Zmax.max()

    @Lazy
    def sightlines(self):
        config = self.config
        sightlinedtype = [('x1', ('f8', 3)),
                       ('x2', ('f8', 3)),
                       ('Z', 'f8'),
                       ('Zmax', 'f8'),
                       ('Zmin', 'f8')]
        if self.Geometry == 'Sphere':
            try:
                linesfile = config.get("IO", 'sightlines')
                raw = numpy.loadtxt(linesfile, usecols=(4, 5, 6, 0), 
                      dtype=[('RA', 'f8'), 
                             ('DEC', 'f8'), 
                             ('Z_VI', 'f8'), 
                             ('THINGID', 'i8')]).view(numpy.recarray)
            except ConfigParser.NoOptionError:
                linesfile = self.datadir + '/QSOcatelog.txt'
                raw = numpy.loadtxt(linesfile, usecols=(0, 1, 2), 
                      dtype=[('RA', 'f8'), 
                             ('DEC', 'f8'), 
                             ('Z_VI', 'f8')
                             ]).view(numpy.recarray)
            sightlines = numpy.empty(raw.size, 
                    dtype=sightlinedtype).view(numpy.recarray)
            sightlines.Z = raw.Z_VI
            sightlines.Zmax = (raw.Z_VI + 1) * 1216. / 1216 - 1
            sightlines.Zmin = (raw.Z_VI + 1) * 1026. / 1216 - 1
            sightlines.Zmin[sightlines.Zmin<self.Redshift] = self.Redshift
            sightlines.Zmax[sightlines.Zmax<self.Redshift] = self.Redshift
            raw.RA *= numpy.pi / 180
            raw.DEC *= numpy.pi / 180
            sightlines.x1[:, 0] = numpy.cos(raw.RA) * numpy.cos(raw.DEC)
            sightlines.x1[:, 1] = numpy.sin(raw.RA) * numpy.cos(raw.DEC)
            sightlines.x1[:, 2] = numpy.sin(raw.DEC)
            sightlines.x2[...] = sightlines.x1
            Dcmin = self.cosmology.Dc(1 / (sightlines.Zmin + 1))[:, None]
            Dcmax = self.cosmology.Dc(1 / (sightlines.Zmax + 1))[:, None]

            sightlines.x1 *= Dcmin
            sightlines.x2 *= Dcmax
            sightlines.x1 *= self.DH
            sightlines.x2 *= self.DH
             
            sightlines.x1 += self.BoxSize * 0.5
            sightlines.x2 += self.BoxSize * 0.5
        elif self.Geometry == 'Test':
            sightlines = numpy.empty(Npixel ** 2, 
                    dtype=sightlinedtype).view(numpy.recarray)
            Zmin = self.Redshift
            Dcmin = self.cosmology.Dc(1 / (Zmin + 1))
            Zmax = 1 / self.cosmology.aback(Dcmin + self.BoxSize / self.DH) - 1
            sightlines.Zmax[...] = Zmax
            sightlines.Zmin[...] = Zmin
            x, y = \
                    numpy.indices((self.Npixel, self.Npixel)).reshape(2, -1) \
                    * (self.BoxSize / self.Npixel)
            sightlines.x1[:, 0] = x
            sightlines.x1[:, 1] = y
            sightlines.x1[:, 2] = 0
            sightlines.x2[...] = sightlines.x1 
            sightlines.x2[:, 2] = self.BoxSize

        assert (sightlines.x1 + 1>= 0).all()
        assert (sightlines.x1 - 1<= self.BoxSize).all()
        assert (sightlines.x2 + 1>= 0).all()
        assert (sightlines.x2 - 1<= self.BoxSize).all()

        return sightlines


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
