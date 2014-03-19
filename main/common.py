import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy
import os.path
import StringIO
import ConfigParser
import argparse
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from lib.cosmology import Cosmology, Lazy


bitmapdtype = numpy.dtype([
    ('objectid', 'i4'),
    ('lambda', 'f4'), 
    ('Z', 'f4'), 
    ('delta', 'f4'), 
    ('F', 'f4'), 
    ('Freal', 'f4'), 
    ('pos', ('f4', 3)), 
    ])

"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("paramfile", 
               help="the paramfile")

    parser.add_argument("command", choices=[
       'sightlines',
       'gaussian',
       'convolve', 
       'matchmeanF',
       'makespectra',
       # above is the pipiline
       'measuremeanF',
       'export',
       'exportross',
       'check',
       # below are (cross)correlation
       'qsocorr',
       'qsocorr2d',
       'pixelcorr',
       'pixelcorr2d',
       'crosscorr2d',
       'corrbootstrap',
       'fit', # will just use the cosmology need to move this thing out
       'testdist',
       ])
"""

class Config(object):
    def export(self, dict, names):
        for name in names:
            setattr(self, name, dict[name])

    def __init__(self, paramfile):
        str = file(paramfile).read().replace(';', ',').replace('#', ';')
        config = ConfigParser.ConfigParser()
        config.readfp(StringIO.StringIO(str))

        self.config = config

        Sigma8 = config.getfloat("Cosmology", "Sigma8")
        OmegaM = config.getfloat("Cosmology", "OmegaM")
        OmegaB = config.getfloat("Cosmology", "OmegaB")
        OmegaL = config.getfloat("Cosmology", "OmegaL")
        BoxPadding = 2000
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

        Seed = config.getint("IC", "Seed")
        NmeshCoarse = config.getint("IC", "NmeshCoarse")
        NmeshEff = config.getint("IC", "NmeshEff")
        Nrep = config.getint("IC", "Nrep")
        Zmin = config.getfloat("IC", "Zmin")
        Zmax = config.getfloat("IC", "Zmax")
        BoxSize = cosmology.Dc(1 / (1 + Zmax)) * DH * 2 + BoxPadding * 2

        KSplit = 0.5 * 2 * numpy.pi / BoxSize * NmeshCoarse * 1.0

        assert NmeshEff % Nrep == 0
        assert NmeshEff % NmeshCoarse == 0

        NmeshFine = NmeshEff / Nrep

        RNG = numpy.random.RandomState(Seed)
        SeedTable = RNG.randint(1<<21 - 1, size=(Nrep,) * 3)

        self.export(locals(), [
            'Seed', 'RNG', 'SeedTable', 'BoxSize', 'Zmin', 'Zmax', 'NmeshFine',
            'NmeshCoarse', 'KSplit', 'BoxPadding',
            'NmeshEff', 'Nrep'])

        print 'BoxSize is', BoxSize
        print 'NmeshFine is', NmeshFine 
        print 'Nrep is', Nrep, 'grid', BoxSize / Nrep
        print 'NmeshEff is', NmeshEff, 'grid', BoxSize / NmeshEff
        print 'NmeshCoarse is', NmeshCoarse, 'grid', BoxSize / NmeshCoarse
        print 'KSplit is', KSplit, 'in realspace', 2 * numpy.pi / KSplit

        QSOScale = config.getfloat("FGPA", "QSOscale")
        Beta = config.getfloat("FGPA", "Beta")
        LogNormalScale = config.getfloat("FGPA", "LogNormalScale")
        Lambda0 = config.getfloat("FGPA", "Lambda0")
        # a pixel will be grid to grid. (grids are the edges)
        LogLamGrid = numpy.log10(Lambda0) + numpy.arange(3000) * 1e-4
        MeanFractionA = config.getfloat("FGPA", "MeanFractionA")
        MeanFractionB = config.getfloat("FGPA", "MeanFractionB")

        VarFractionA = config.getfloat("FGPA", "VarFractionA")
        VarFractionB = config.getfloat("FGPA", "VarFractionB")
        VarFractionZ = config.getfloat("FGPA", "VarFractionZ")

        IGMTemperature = config.getfloat('FGPA', 'IGMTemperature')

        NmeshLyaBox = 2 ** (int(numpy.log2(BoxSize / NmeshEff / LogNormalScale) + 0.5))
        NmeshQSO = 2 ** (int(numpy.log2(BoxSize / Nrep / QSOScale) + 0.5))
        if NmeshQSO < 1: NmeshQSO = 1

        NmeshQSOEff = NmeshQSO * Nrep
        print 'NmeshLyaBox is ', NmeshLyaBox, 'grid', BoxSize / NmeshEff / NmeshLyaBox
        print 'NmeshQSO is', NmeshQSO, 'grid', BoxSize / NmeshQSOEff

        self.export(locals(), [
            'Beta', 'LogNormalScale',
            'NmeshLyaBox', 'QSOScale', 'NmeshQSO', 'NmeshQSOEff', 
            'IGMTemperature', 'LogLamGrid'] )

        self.export(locals(), [
        'MeanFractionA',
        'MeanFractionB',
        'VarFractionA',
        'VarFractionB',
        'VarFractionZ',
        ])


        datadir = config.get("IO", "datadir")
        try:
            QSOCatelog = config.get("IO", "QSOCatelog")
        except ConfigParser.NoOptionError:
            QSOCatelog = datadir + '/QSOCatelog.raw'
        try:
            DeltaField = config.get("IO", "DeltaField")
        except ConfigParser.NoOptionError:
            DeltaField = datadir + '/DeltaField.raw'
        try:
            ObjectIDField = config.get("IO", "ObjectIDField")
        except ConfigParser.NoOptionError:
            ObjectIDField = datadir + '/ObjectIDField.raw'
        try:
            VelField = config.get("IO", "VelField")
        except ConfigParser.NoOptionError:
            VelField = datadir + '/VelField.raw'

        try:
            PowerSpectrum = config.get("Cosmology", "PowerSpectrum")
        except ConfigParser.NoOptionError:
            PowerSpectrum = datadir + '/power.txt'

        self.export(locals(), ['datadir', 'QSOCatelog', 'PowerSpectrum', 
                'DeltaField', 'ObjectIDField', 'VelField'])

        try:
            SpectraOutputTauRed = config.get("Cosmology", "SpectraOutputTauRed")
        except ConfigParser.NoOptionError:
            SpectraOutputTauRed = datadir + '/spectra-taured.raw'

        try:
            SpectraOutputTauReal = config.get("Cosmology", "SpectraOutputTauReal")
        except ConfigParser.NoOptionError:
            SpectraOutputTauReal = datadir + '/spectra-taureal.raw'

        try:
            SpectraOutputDelta = config.get("Cosmology", "SpectraOutputDelta")
        except ConfigParser.NoOptionError:
            SpectraOutputDelta = datadir + '/spectra-delta.raw'

        self.export(locals(), [
                'SpectraOutputTauRed',
                'SpectraOutputTauReal',
                'SpectraOutputDelta',
                ])

    @Lazy
    def QSObias(self):
        """ returns a function evaluating QSO bias
            at given R

            current file is copied from 
        """
        config = self.config
        string = """
           0.24   1.41   0.18
           0.49   1.38   0.06
           0.80   1.45   0.38
           1.03   1.83   0.03
           1.23   2.37   0.25
           1.41   1.92   0.50
           1.58   2.42   0.40
           1.74   2.79   0.47
           1.92   3.62   0.49
           2.10   2.99   1.42
           2.40   4      5
           3.20   6      2
           4.0    10.5     3
        """
        try:
            biasfile = config.get("QSO", "biasfile")
            z, bias, err = numpy.loadtxt(biasfile, unpack=True)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            z, bias, err = numpy.fromstring(string, sep=' ').reshape(-1, 3).T
        a = 1 / (z + 1.)
        R = self.cosmology.Dc(a) * self.DH
        spl = UnivariateSpline(R, bias, 1 / err)
        return spl 

    @Lazy
    def QSOdensity(self):
        """ returns a function evaluating 
            mean QSO density at given R
            the input QSOdensityfile, constains
            two npy arrays, 
            bins, the zbins (N + 1 entries)
            count, the number count of QSOs in that bin, 
            (N entries)

            This file can be prepared with 
            f = gaepsi.cosmology.agn.qlf_observer.
            and output f(z)
        """
        string = """
0.1 5.20571573185e-15 0.2 2.96627124784e-14
0.4 5.94935402372e-14 0.6 6.02256554399e-14
0.8 5.30889459081e-14 1.0 4.79606024125e-14
1.2 4.37453636884e-14 1.4 3.98509187531e-14
1.6 3.58656511045e-14 1.8 3.16185604541e-14
2.0 2.71388657143e-14 2.2 2.25732139999e-14
2.4 1.81159328576e-14 2.6 1.39619212926e-14
2.8 1.02769869655e-14 3.0 7.17907201412e-15
3.2 4.72732379062e-15 3.4 2.91919547441e-15
3.6 1.69562293098e-15 3.8 9.54231503902e-16
4.0 5.68664902877e-16 4.2 4.10471722414e-16
4.4 3.68932895967e-16 4.6 3.64347275462e-16
4.8 3.52067723866e-16 5.0 3.17466641007e-16
5.2 2.60245877731e-16 5.4 1.96528168034e-16
5.6 1.34140178176e-16 5.8 8.16047392053e-17
6.0 4.37813317973e-17 """
        try:
            QSOdensity = self.config.get("QSO", "QSOdensityfile")
            z = numpy.load(QSOdensity)['z']
            density = numpy.load(QSOdensity)['density']
            print 'using ', QSOdensity, 'for mean qso density'
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            print 'using builtin i band limited hopkins qlf for mean qso density'
            z, density = numpy.fromstring(string, sep=' ').reshape(-1, 2).T
        a = 1 / (z + 1.)
        R = self.cosmology.Dc(a) * self.DH
        return interp1d(R, density, bounds_error=False,
                fill_value=0.0)


    @Lazy
    def SurveyQSOdensity(self):
        """ returns a function evaluating 
            Survey QSO number density at given R
            This will be adjusted by the sky mask

            the input surveydensityfile, constains
            two npy arrays, 
            z
            specdensity, dN / dz where N is number QSOs

            The default is to use the SDSS DR9 count.
        """
        string = """
0.0 34.0985107917 0.2 959.763566767 0.4 6605.80243392 0.6 32523.7675186
0.8 72691.414628 1.0 26855.6444122 1.2 11530.0423145 1.4 12669.611195
1.6 25099.5717001 1.8 22712.9437337 2.0 35654.3477467 2.2 153115.318974
2.4 178890.569673 2.6 115889.244553 2.8 69022.9874509 3.0 56040.3877876
3.2 41957.1799842 3.4 18310.9661606 3.6 11673.2289537 3.8 9949.67508789
4.0 5060.53773401 4.2 2350.06892613 4.4 1150.65036452 4.6 767.760735547
4.8 603.559006041 5.0 266.067138644 5.2 108.856505818 5.4 18.8484273619
5.6 5.94102761023 5.8 4.30473603211 6.0 0.662939420386 
"""
        try:
            QSOcount = self.config.get("QSO", "Surveydensityfile")
            z = numpy.load(QSOdensity)['z']
            specdensity = numpy.load(QSOdensity)['specdensity']
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            print 'using builtin DR9 qso specdensity file'
            z, specdensity = numpy.fromstring(string, sep=' ').reshape(-1, 2).T
            
        a = 1 / (z + 1)
        R = self.cosmology.Dc(a) * self.DH
        density = specdensity * (- a ** -2) * \
        self.cosmology.aback(R / self.DH, nu=1) / self.DH / (4 * numpy.pi * R ** 2)
        # adjust for sky 
        density /= self.skymask.fraction
        density[0] = 0
        return interp1d(R, density, bounds_error=False, fill_value=0.0)

    @Lazy
    def skymask(self):
        """ sky mask in mango healpix format """
        try:
            import chealpy
        except ImportError:
            print "chealpy is not installed, skymask is disabled"
            def func(xyz):
                return numpy.ones(shape=xyz.shape[0])
            func.Nside = 4
            func.fraction = 1.0
            func.mask = numpy.ones(4)
            return func 
        try:
            skymask = self.config.get("QSO", "skymaskfile")
            skymask = numpy.loadtxt(skymask, skiprows=1)
        except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
            skymask = numpy.ones(2 * 2 * 12)
        Nside = chealpy.npix2nside(len(skymask))
        print 'using healpix sky mask Nside', Nside
        def func(xyz):
            """ look up the sky mask from xyz vectors
                xyz is row vectors [..., 3] """
            ipix = chealpy.vec2pix_nest(Nside, xyz)
            #Nside, 0.5 * numpy.pi - dec, ra)
            return skymask[ipix]

        func.Nside = Nside
        func.fraction = skymask.sum() / len(skymask)
        func.mask = skymask
        return func 
         
    def yieldwork(self):
        for i in range(self.Nrep):
            for j in range(self.Nrep):
                for k in range(self.Nrep):
                    yield i, j, k
    @Lazy
    def fibers(self):
        fibers = self.config.get("IO", 'fibers')
        array = numpy.loadtxt(fibers, usecols=(1, 2, 3, 4, 5, 6),
            dtype=[
                ('PLATE', 'i2'),
                ('MJD', 'i4'),
                ('FIBERID', 'i2'),
                ('RA', 'f8'),
                ('DEC', 'f8'),
                ('Z_VI', 'f8')])
        fibers = array[array['Z_VI'].argsort()]
        return fibers
    def xyz2dc(self, xyz):
        """ returns the comoving distance of position xyz,
            xyz is in full box box units. (not any grids),
            return value is in cosmology units (DH)
        """
        R = ((xyz) ** 2).sum(axis=-1) ** 0.5
        R /= self.DH
        return R

    def layout(self, Ntot, chunksize):
        layout = Layout(Nrep=self.Nrep, Ntot=Ntot, chunksize=chunksize)
        layout.REPoffset = \
                numpy.array(numpy.indices((self.Nrep,) * 3)).transpose((1, 2, 3, 0))
        layout.origin = \
               1.0 * layout.REPoffset * self.BoxSize / self.Nrep - self.BoxSize * 0.5
        return layout

class Chunk(object):
    def __init__(self, box, i):
        self.i = i 
        self.box = box
        self._dict = {}
        self.offset = i * box.layout.chunksize

    def getslice(self):
        return slice(self.offset, self.offset + self.box.layout.chunksize)
    def __getattr__(self, attr):
        return getattr(self.box, attr)[self.i:self.i+1].squeeze()
    def __getitem__(self, item):
        return self._dict[item]
    def __setitem__(self, item, value):
        self._dict[item] = value

class Box(object):
    def __init__(self, layout, i, j, k):
        self.i, self.j, self.k = i, j, k
        self.layout = layout
    def __getitem__(self, i):
        return Chunk(self, i)
    def __getattr__(self, attr):
        return getattr(self.layout, attr)[self.i, self.j, self.k]
    def __iter__(self):
        for i in range(self.layout.Nchunks):
            yield self[i]
    
class Layout(object):
    def __init__(self, Nrep, Ntot, chunksize):
        self.Nrep = Nrep
        self.Ntot = Ntot
        self.chunksize = chunksize
        self.Nchunks = \
            (Ntot + chunksize - 1) // chunksize
    def __getitem__(self, index):
        return Box(self, *index)
    def __iter__(self):
        for i in range(self.Nrep):
            for j in range(self.Nrep):
                for k in range(self.Nrep):
                    yield self[i, j, k]
def PowerSpectrum(A):
    power = A.PowerSpectrum
    try:
        k, p = numpy.loadtxt(power, unpack=True)
        print 'using power from file ', power
    except IOError:
        print 'using power from pycamb, saving to ', power
        Pk = A.cosmology.Pk
        k = Pk.x / A.DH
        p = Pk.y * A.DH ** 3 * (2 * numpy.pi) ** -3
        numpy.savetxt(power, zip(k, p))

    p[numpy.isnan(p)] = 0
    power = k, p
    return power

def MeanFractionModel(config):
    A = config.MeanFractionA
    B = config.MeanFractionB
    def func(a):
        return numpy.exp(A * a ** -B)
    return func

def VarFractionModel(config):
    A = config.VarFractionA
    B = config.VarFractionB
    Z = config.VarFractionZ
    mfm = MeanFractionModel(config)
    def func(a):
        return A * (1 / a / Z) ** B * mfm(a) ** 2
    return func

if __name__ == '__main__':
    from sys import argv
    config = Config(argv[1])

"""
    @Lazy
    def sightlines(self):
        config = self.config
        internalsightlinedtype = [('x1', ('f8', 3)),
                       ('x2', ('f8', 3)),
                       ('dir', ('f8', 3)),
                       ('Zreal', 'f8'),
                       ('Z', 'f8'),
                       ('Zmax', 'f8'),
                       ('Zmin', 'f8'),
                       ('Rmax', 'f8'),
                       ('Rmin', 'f8'),
                       ('RA', 'f8'),
                       ('DEC', 'f8'),
                       ('refMJD', 'i4'),
                       ('refPLATE', 'i2'),
                       ('refFIBERID', 'i2'),
                       ]
        try:
            linesfile = config.get("IO", 'sightlines')
            raw = numpy.loadtxt(linesfile, usecols=(4, 5, 6, 0), 
                  dtype=[('RA', 'f8'), 
                         ('DEC', 'f8'), 
                         ('Z_VI', 'f8'), 
                         ('THINGID', 'i8'),
                         ('PLATE', 'i2'),
                         ('MJD', 'i4'),
                         ('FIBERID', 'i2'),
                         ]).view(numpy.recarray)
        except ConfigParser.NoOptionError:
            linesfile = self.datadir + '/QSOcatelog.raw'
            raw = numpy.fromfile(linesfile, dtype=sightlinedtype)\
                    .view(numpy.recarray)
        sightlines = numpy.empty(raw.size, 
                dtype=internalsightlinedtype).view(numpy.recarray)
        sightlines.RA = raw.RA
        sightlines.DEC = raw.DEC
        sightlines.refPLATE = raw.refPLATE
        sightlines.refFIBERID = raw.refFIBERID
        sightlines.refMJD = raw.refMJD
        sightlines.Z = raw.Z_VI
        sightlines.Zreal= raw.Z_REAL
        Zmax = (raw.Z_REAL + 1) * 1216. / 1216 - 1
        Zmin = (raw.Z_REAL + 1) * 1026. / 1216 - 1
        Zmin[sightlines.Zmin<0] = 0
        Zmax[sightlines.Zmax<0] = 0
        raw.RA *= numpy.pi / 180
        raw.DEC *= numpy.pi / 180
        sightlines.x1[:, 0] = numpy.cos(raw.RA) * numpy.cos(raw.DEC)
        sightlines.x1[:, 1] = numpy.sin(raw.RA) * numpy.cos(raw.DEC)
        sightlines.x1[:, 2] = numpy.sin(raw.DEC)
        sightlines.x2[...] = sightlines.x1
        # a bit of padding for RSD
        sightlines.Rmin = self.cosmology.Dc(1 / (Zmin + 1))\
                * self.DH - self.BoxPadding
        sightlines.Rmax = self.cosmology.Dc(1 / (Zmax + 1)) \
                * self.DH + self.BoxPadding
        sightlines.x1 *= sightlines.Rmin[:, None]
        sightlines.x2 *= sightlines.Rmax[:, None]
        sightlines.x1 += self.BoxSize * 0.5
        sightlines.x2 += self.BoxSize * 0.5

        sightlines.dir = sightlines.x2 - sightlines.x1
        sightlines.dir *= numpy.einsum('ij, ij->i', sightlines.dir,
                sightlines.dir)[:, None] ** -0.5

        if (sightlines.x1 + 1>= 0).all() and \
           (sightlines.x1 - 1<= self.BoxSize).all() and \
           (sightlines.x2 + 1>= 0).all() and \
           (sightlines.x2 - 1<= self.BoxSize).all():
            pass
        else:
            print 'some sightlines are not fully in box'

        return sightlines

    def sightlinebins(self, sightline):
        Zqso = sightline.Zqso
        bins = numpy.arange(
            sightline.Rmin[i],
            sightline.Rmax[i] + self.PixelScale,
            self.PixelScale)
        return bins
"""


