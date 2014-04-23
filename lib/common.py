import numpy
import os.path
import StringIO
import ConfigParser
import argparse
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad, simps
from lib.cosmology import Cosmology, Lazy
from lib.sph_bessel import j0, j2, j4

class ConfigBase(object):
    def __init__(self, paramfile):
        s = file(paramfile).read().replace(';', ',').replace('#', ';')
        config = ConfigParser.ConfigParser()
        config.readfp(StringIO.StringIO(s))

        self.config = config
        export = self.export

        export("Cosmology", "G", type=float, default=43007.1)
        export("Cosmology", "C", type=float, default=299792.458)
        export("Cosmology", "H0", type=float, default=0.1)
        export("Cosmology", "LymanAlpha", type=float, default=1216.0)
        export("Cosmology", "LymanBeta", type=float, default=1026.0)


        self.DH = self.C / self.H0

        export("Cosmology", [
            "Sigma8",
            "OmegaM",
            "OmegaB",
            "OmegaL", 
            "h"] , type=float)

        self.cosmology = Cosmology(M=self.OmegaM, 
            L=self.OmegaL, B=self.OmegaB, h=self.h, sigma8=self.Sigma8)

    def export(self, section, names, type=str, **kwargs):
        if not isinstance(names, (list, tuple)):
            names = [names, ]

        for name in names:
            try:
                s = self.config.get(section, name)
                v = type(s)
            except ConfigParser.NoOptionError:
                if 'default' in kwargs:
                    v = kwargs['default']
                else:
                    raise
            setattr(self, name, v)

class Config(ConfigBase):
    def __init__(self, paramfile, basedir=None):
        """ if given basedir will be prefixed to datadir """
        ConfigBase.__init__(self, paramfile)

        export = self.export

        export("IC", [
            "Seed",
            "NmeshCoarse",
            "NmeshEff",
            "Nrep"], type=int)

        export("IC", ['Zmin', 'Zmax'], type=float)
        export("IC", "BoxPadding", type=float, default=2000.)


        self.BoxSize = self.cosmology.Dc(1 / (1 + self.Zmax)) * self.DH * 2 + self.BoxPadding * 2
        self.KSplit = 0.5 * 2 * numpy.pi / self.BoxSize * self.NmeshCoarse * 1.0

        assert self.NmeshEff % self.Nrep == 0
        assert self.NmeshEff % self.NmeshCoarse == 0

        self.NmeshFine = self.NmeshEff / self.Nrep

        self.RNG = numpy.random.RandomState(self.Seed)
        self.SeedTable = self.RNG.randint(1<<21 - 1, size=(self.Nrep,) * 3)

        # a good value of LogNormalScale is 250 Kpc/h, which gives
        # a sigma of ~ 1.0 at z=3.0, agreeing with Bi & Davidson 1997.

        export("FGPA", [
            "LogNormalScale",
            "Lambda0",
            "MeanFractionA",
            "MeanFractionB",
            "VarFractionA",
            "VarFractionB",
            "VarFractionZ",
            "IGMTemperature",
            ], type=float)

        # a pixel will be grid to grid. (grids are the edges)
        self.LogLamGrid = numpy.log10(self.Lambda0) + numpy.arange(5000) * 1e-4

        # make sure it is smaller than the LogNormalScale
        self.NmeshLyaBox = 2 ** (int(numpy.log2(self.BoxSize / self.NmeshEff / self.LogNormalScale) + 1))


        export("Quasar", 
                [
                    "QSOBiasInput",
                    "QSODensityInput",
                    "SkymaskInput",
                ] , 
                default=None)

        export("Quasar", "QSOScale", type=float)

#        self.NmeshQSO = 2 ** (int(numpy.log2(self.BoxSize / self.Nrep / self.QSOScale) + .5))
#        if self.NmeshQSO < 1: self.NmeshQSO = 1
        self.NmeshQSO = int(self.BoxSize / self.Nrep / self.QSOScale)
        if self.NmeshQSO < 1: self.NmeshQSO = 1

        export("Output", "datadir")
        if basedir is not None:
            self.datadir = os.path.join(basedir, self.datadir)

        jn = lambda x: os.path.join(self.datadir, x)
        export("Output", "PowerSpectrumCache", default=jn('power.txt'))
        export("Output", "QSOCatelog", default=jn('QSOCatelog.raw'))
        export("Output", "DeltaField", default=jn('DeltaField.raw'))
        export("Output", "ObjectIDField", default=jn('ObjectIDField.raw'))
        export("Output", "VelField", default=jn('VelField.raw'))
        export("Output", "MatchMeanFractionOutput", default=jn('MatchMeanFractionOutput.npz'))
        export("Output", "MeasuredMeanFractionOutput", default=jn('MeasuredMeanFractionOutput.npz'))
        export("Output", "SpectraOutputTauRed", default=jn('SpectraOutputTauRed.raw'))
        export("Output", "SpectraOutputTauReal", default=jn('SpectraOutputTauReal.raw'))
        export("Output", "SpectraOutputDelta", default=jn('SpectraOutputDelta.raw'))
        print str(self)

    def __str__(self):
        s = """
        BoxSize = %(BoxSize)g
        NmeshFine = %(NmeshFine)d
        Nrep = %(Nrep)d
        NmeshCoarse = %(NmeshCoarse)d
        KSplit = %(KSplit)f
        NmeshLyaBox = %(NmeshLyaBox)d
        NmeshQSO = %(NmeshQSO)d
        """
        return s % self.__dict__


    def yieldwork(self):
        for i in range(self.Nrep):
            for j in range(self.Nrep):
                for k in range(self.Nrep):
                    yield i, j, k

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

def QSOBiasModel(config):
    """ returns a function evaluating QSO bias
        at given a

        current file is copied from 
    """
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
    if config.QSOBiasInput is None:
        z, bias, err = numpy.fromstring(string, sep=' ').reshape(-1, 3).T
    else:
        z, bias, err = numpy.loadtxt(config.QSOBiasInput, unpack=True)
    a = 1 / (z + 1.)
#    R = config.cosmology.Dc(a) * config.DH
    spl = UnivariateSpline(a, bias, 1 / err)
    return spl 

def QSODensityModel(config):
    """ returns a function evaluating 
        Survey QSO number density at given a
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
    skymask = Skymask(config)
    if config.QSODensityInput is None:
        z, specdensity = numpy.fromstring(string, sep=' ').reshape(-1, 2).T
    else:
        z, specdensity = numpy.loadtxt(config.QSODensityInput, unpack=True)
        
    # adjust for sky 
    specdensity = specdensity / skymask.fraction

    # now lets convert it to number density per comoving volume
    a = 1 / (z + 1)
    R = config.cosmology.Dc(a) * config.DH
    # dN/dV = 1 / 4piR^2 dN/dR
    # dN/dR = dN/dz * dz/da * da/dR
    density = specdensity * (- a ** -2) * \
    config.cosmology.aback(R / config.DH, nu=1) / config.DH / (4 * numpy.pi * R ** 2)

    # fix it at z=0.
    density[0] = 0
    rt = interp1d(a[::-1], density[::-1], bounds_error=False, fill_value=0.0)
    mask = (z >= config.Zmin) & (z <= config.Zmax)
    rt.Nqso = numpy.trapz(specdensity[mask], x=z[mask])
    return rt
try:
    import chealpy
except ImportError:
    chealpy = None
    print "chealpy is not installed, skymask is disabled"

class Skymask(object):
    """ sky mask in mango healpix format """
    def __init__(self, config):
        if config.SkymaskInput is not None:
            self.Nside = 2
            self.fraction = 1.0
            self.mask = numpy.ones(48)
        else:
            if chealpy is None:
                raise Exception("Skymask is provided yet chealpy is not installed")
            skymask = numpy.loadtxt(config.SkymaskInput, skiprows=1)
            Nside = chealpy.npix2nside(len(skymask))
            print 'using healpix sky mask Nside', Nside
            self.Nside = Nside
            self.fraction = skymask.sum() / len(skymask)
            self.mask = skymask

    def __call__(self, xyz):
        """ look up the sky mask from xyz vectors
            xyz is row vectors [..., 3] """
        if chealpy is None:
            # it is guarranted config.SkymaskInput is None
            return numpy.ones(shape=xyz.shape[0])
        else:
            ipix = chealpy.vec2pix_nest(self.Nside, xyz)
            #Nside, 0.5 * numpy.pi - dec, ra)
            return self.mask[ipix]

class PowerSpectrum(object):
    """ PowerSpectrum is camb at z=0. Need to use
        Dplus to grow(shrink) to given redshift
    """
    def __init__(self, A, smoothingscale=0.0):
        power = A.PowerSpectrumCache
        try:
            k, p = numpy.loadtxt(power, unpack=True)
            print 'using power from file ', power
        except IOError:
            print 'using power from pycamb, saving to ', power
            Pk = A.cosmology.Pk
            k = Pk.x / A.DH
            p = Pk.y * A.DH ** 3 * (2 * numpy.pi) ** -3
            numpy.savetxt(power, zip(k, p))

        # k ,and p are in KPC/h units!
        p[numpy.isnan(p)] = 0
        self.k = k
        self.p = p
        self.p *= numpy.exp(- (k * smoothingscale) ** 2)
    def __getitem__(self, i):
        return [self.k, self.p][i]
    def __iter__(self):
        return iter([self.k, self.p])

    def monopole(self, r):
        return self.pole(r, 0)

    def quadrupole(self, r):
        return self.pole(r, 2)

    def hexadecapole(self, r):
        return self.pole(r, 4)

    def __call__(self, K):
        return self.Pfunc(K)

    @Lazy
    def Pfunc(self):
        K, P = self
        mask = ~numpy.isnan(P) & (K > 0) & (P > 0)
        K = K[mask]
        P = P[mask]
        intp = interp1d(numpy.log10(K), numpy.log10(P), kind=5, fill_value=-99, bounds_error=False)
        def func(k):
            return 10 **intp(numpy.log10(k))
        return func
    def pole(self, r, order):
        """calculate the multipoles of xi at order for given P(K)
            in kpc/h units 
            note that the j2 pole is negative from pat's
            1104.5244v2.pdf
        """
        assert order in [0, 2, 4]
        kernel = [j0, None, j2, None, j4][order]
        Pfunc = self.Pfunc
        xi = numpy.empty_like(r)
        K, P = self
        Kmin = K.min()
        Kmax = K.max()
        for i in range(len(r)):
            def func(k):
                # damping from Xiao
                rt = 4 * numpy.pi * Pfunc(k) * k ** 2 * kernel(k * r[i])
                rt = rt * (2 * numpy.pi) ** 3 # going from GADGET to xiao
                return rt
            xi[i] = quad(func, Kmin, Kmax, limit=400, limlst=400, maxp1=400)[0]
        return xi * (2 * numpy.pi) ** -3

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

def MeanFractionMeasured(config, real=False):
    f = numpy.load(config.MeasuredMeanFractionOutput)
    ameanFbins = f['abins']
    if real:
        meanF = f['xmeanFreal']
    else:
        meanF = f['xmeanF']
    def func(a):
        dig = numpy.digitize(a, ameanFbins).clip(0, len(ameanFbins) - 2)
        return meanF[dig]

    return func

class FGPAmodel(object):
    # turns out Af is exp(u * a + v), 
    # and Bf is u * a **2 + v * a + w.
    # thus we use polyfit in FGPAmodel
    def __init__(self, config):
        f = numpy.load(config.MatchMeanFractionOutput)
        a = f['a']
        Af = f['Af']
        Bf = f['Bf']
        arg = a.argsort()
        Af = Af[arg]
        Bf = Bf[arg]
        a = a[arg]
        # reject bad fits
        mask = (Af > 0)
        # skip the first bin because it
        # can be very much off (due to small sample size).
        self.a = a[mask][1:]
        self.Af = Af[mask][1:]
        self.Bf= Bf[mask][1:]
        
    @Lazy
    def Afunc(self):
        pol = numpy.polyfit(self.a, numpy.log(self.Af), 1)
        def func(a):
            return numpy.exp(numpy.polyval(pol, a))
        return func
    @Lazy
    def Bfunc(self):
        pol = numpy.polyfit(self.a, self.Bf, 2)
        def func(a):
            return numpy.polyval(pol, a)
        return func


class Sightlines(object):
    dtype =numpy.dtype([('RA', 'f8'), 
                             ('DEC', 'f8'), 
                             ('Z_RED', 'f8'),
                             ('Z_REAL', 'f8'),
                             ])

    def __init__(self, config, LogLamMin=None, LogLamMax=None):
        """ create a sightline catelogue.

            if LogLamMin and LogLamMax are given,
            each sightline is chopped at these wave lengths.

            otherwise the lines will cover from config.LymanBeta to config.LymanAlpha.   
        """

        self.data = numpy.fromfile(config.QSOCatelog, 
                dtype=Sightlines.dtype)
        self.Z_REAL = self.data['Z_REAL']
        self.DEC = self.data['DEC']
        self.RA = self.data['RA']
        self.config = config
        self.LogLamMin = numpy.log10((self.Z_REAL + 1) * self.config.LymanBeta)

        if LogLamMin is not None:
            self.LogLamMin = numpy.maximum(LogLamMin, self.LogLamMin)

        self.LogLamMax = numpy.log10((self.Z_REAL + 1) * config.LymanAlpha)
        if LogLamMax is not None:
            self.LogLamMax = numpy.minimum(LogLamMax, self.LogLamMax)

        # Z_RED is writable!
        data2 = numpy.memmap(config.QSOCatelog, dtype=Sightlines.dtype, mode='r+')
        self.Z_RED = data2['Z_RED']

    def __len__(self):
        return len(self.data)

    @Lazy
    def R(self):
        cosmology = self.config.cosmology
        a2 = 1. / (self.Z_RED + 1)
        R2 = cosmology.Dc(a2) * self.config.DH
        return R2

    @Lazy
    def ActiveSampleStart(self):
        """ the sample layout is decided from LambdaMin / LambdaMax;
            this is a safe offset rel to SampleOffset,
            and with added padding so that thermal convolution is safe.
        """
        cosmology = self.config.cosmology
        a1 = self.config.LymanAlpha / 10 ** self.LogLamMin 
        R1 = cosmology.Dc(a1) * self.config.DH
        R1full = self.R1
        rel = numpy.int32((R1 - R1full - 4000) // self.config.LogNormalScale)
        rel[rel < 0] = 0
        big = rel > self.Nsamples
        rel[big] = self.Nsamples[big]
        return rel

    @Lazy
    def ActiveSampleEnd(self):
        cosmology = self.config.cosmology
        a2 = self.config.LymanAlpha / 10 ** self.LogLamMax
        R2 = cosmology.Dc(a2) * self.config.DH
        R1full = self.R1
        rel = numpy.int32((R2 - R1full + 4000) // self.config.LogNormalScale)
        rel[rel < 0] = 0
        big = rel > self.Nsamples
        rel[big] = self.Nsamples[big]
        return rel

    @Lazy
    def SampleOffset(self):
        """ the full sample layout is decided from Z_REAL used by gaussian"""
        rt = numpy.empty(len(self), dtype='intp')
        rt[0] = 0
        rt[1:] = numpy.cumsum(self.Nsamples)[:-1]
        return rt

    @Lazy
    def R1(self):
        """ to get the Dc grid in Kpc/h units, use 
            R1 + arange(Nsamples) * config.LogNormalScale
        """
        cosmology = self.config.cosmology
        a1 = self.config.LymanAlpha / (self.config.LymanBeta * (self.Z_REAL + 1))
        R1 = cosmology.Dc(a1) * self.config.DH
        return R1

    @Lazy
    def R2(self):
        cosmology = self.config.cosmology
        a2 = 1. / (self.Z_REAL + 1)
        R2 = cosmology.Dc(a2) * self.config.DH
        return R2

    @Lazy
    def Nsamples(self):
        cosmology = self.config.cosmology
        return numpy.int32((self.R2 - self.R1) / self.config.LogNormalScale)

    @Lazy
    def x1(self):
        cosmology = self.config.cosmology
        return self.dir * self.R1[:, None]

    @Lazy
    def x2(self):
        cosmology = self.config.cosmology
        return self.dir * self.R2[:, None]

    @Lazy
    def dir(self):
        dir = numpy.empty((len(self), 3))
        dir[:, 0] = numpy.cos(self.RA) * numpy.cos(self.DEC)
        dir[:, 1] = numpy.sin(self.RA) * numpy.cos(self.DEC)
        dir[:, 2] = numpy.sin(self.DEC)
        return dir

    @Lazy
    def LogLamGridIndMin(self):
        rt = numpy.searchsorted(self.config.LogLamGrid, 
                    self.LogLamMin, side='right')
        return rt
    @Lazy
    def LogLamGridIndMax(self):
        rt = numpy.searchsorted(self.config.LogLamGrid, 
                    self.LogLamMax, side='left')
        # probably no need to care if it goes out of limit. 
        # really should have used 1e-4 binning than the search but
        # need to deal with the clipping later on.
        # Max is exclusive!
        # AKA the right edge of the last loglam bin is at Max - 1
        # in config.LogLamGrid
        toosmall = rt <= self.LogLamGridIndMin
        rt[toosmall] = self.LogLamGridIndMin[toosmall] + 1
        return rt

    @Lazy
    def Npixels(self):
        # This is always 1 smaller than the number of Bins edges. 
        return self.LogLamGridIndMax - self.LogLamGridIndMin - 1

    def GetPixelLogLamBins(self, i):
        sl = slice(self.LogLamGridIndMin[i], self.LogLamGridIndMax[i])
        bins = self.config.LogLamGrid[sl]
        assert len(bins) == self.Npixels[i] + 1
        return bins

    def GetPixelLogLamCenter(self, i):
        bins = self.GetPixelLogLamBins(i)
        return 0.5 * (bins[1:] + bins[:-1])

    @Lazy
    def PixelOffset(self):
        rt = numpy.empty(len(self), dtype='intp')
        rt[0] = 0
        rt[1:] = numpy.cumsum(self.Npixels)[:-1]
        return rt

class SpectraOutput(object):
    def __init__(self, config):
        self.config = config
        sightlines = Sightlines(config)
        class Accessor(object):
            @staticmethod
            def getslice(index):
                sl = slice(
                    sightlines.PixelOffset[index],
                    sightlines.PixelOffset[index] + 
                    sightlines.Npixels[index])
                return sl
            def __init__(self, data):
                self.data = data
            def __getitem__(self, index):
                return self.data[self.getslice(index)] 
        self.Accessor = Accessor
        class Faker(object):
            def __init__(self, table):
                self.table = table
            def __getitem__(self, index):
                """ table is the central value, IndMax refers to the last edge.
                    hence needs to take one away from IndMax
                """
                sl = slice(
                sightlines.LogLamGridIndMin[index],
                sightlines.LogLamGridIndMax[index] - 1)
                return self.table[sl]
        self.Faker = Faker
        self.sightlines = sightlines
    def __len__(self):
        return len(self.sightlines)
    @Lazy
    def taured(self):
        taured = numpy.memmap(self.config.SpectraOutputTauRed, mode='r+', dtype='f4')
        return self.Accessor(taured) 
        
    @Lazy
    def taureal(self):
        taureal = numpy.memmap(self.config.SpectraOutputTauReal, mode='r+', dtype='f4')
        return self.Accessor(taureal) 

    @Lazy
    def delta(self):
        delta = numpy.memmap(self.config.SpectraOutputDelta, mode='r+', dtype='f4')
        return self.Accessor(delta)

    @Lazy
    def LogLam(self):
        LogLamGrid = self.config.LogLamGrid
        LogLamCenter = 0.5 * (LogLamGrid[1:] + LogLamGrid[:-1])
        return self.Faker(LogLamCenter)

    @Lazy
    def Lam(self):
        LogLamGrid = self.config.LogLamGrid
        LamCenter = 10 ** (0.5 * (LogLamGrid[1:] + LogLamGrid[:-1]))
        return self.Faker(LamCenter)

    @Lazy
    def z(self):
        LogLamGrid = self.config.LogLamGrid
        LogLamCenter = 0.5 * (LogLamGrid[1:] + LogLamGrid[:-1])
        z = 10 ** LogLamCenter / self.config.LymanAlpha - 1
        return self.Faker(z)

    @Lazy
    def a(self):
        LogLamGrid = self.config.LogLamGrid
        LogLamCenter = 0.5 * (LogLamGrid[1:] + LogLamGrid[:-1])
        a = self.config.LymanAlpha / 10 ** LogLamCenter
        return self.Faker(a)
        
    @Lazy
    def R(self):
        LogLamGrid = self.config.LogLamGrid
        LogLamCenter = 0.5 * (LogLamGrid[1:] + LogLamGrid[:-1])
        z = 10 ** LogLamCenter / self.config.LymanAlpha - 1
        a = 1 / (z + 1)
        Dc = self.config.cosmology.Dc(a) * self.config.DH
        return self.Faker(Dc)

    def position(self, i):
        """ returns a vector of the positions of pixels in the spectra """
        return self.R[i][:, None] * self.sightlines.dir[i][None, :]

    def RfLam(self, i):
        return self.Lam[i] / (1 + self.sightlines.Z_RED[i])

from numpy.polynomial.legendre import Legendre, legfit, legvander

class CorrFunc(object): 
    def __init__(self, r, mu, xi=None):
        self.r = r
        self.mu = mu
        if xi is None:
            xi = numpy.zeros((len(r), len(mu)))

        assert xi.shape == (len(r), len(mu))

        self.xi = xi

    @Lazy
    def rmesh(self):
        return numpy.tile(self.r[:, None], (1, len(self.mu)))
    @Lazy
    def mumesh(self):
        return numpy.tile(self.mu[None, :], (len(self.r), 1))

    def frompoles(self, poles):
        v = legvander(self.mu, poles.shape[1] - 1)
        self.xi[...] = numpy.einsum('jl,il->ij', v, poles)
        if hasattr(self, 'poles'):
            del self.poles

    @Lazy
    def poles(self):
        orders = 4
        xi = self.xi
        r = self.r
        mu = self.mu
        
        Nr, Nmu = self.xi.shape
        v = legvander(mu, orders)
        poles = numpy.empty((len(r), orders+1))
        sym = (mu >= 0).all()
        for i in range(orders + 1):
            norm = simps(v[:, i] ** 2, mu)            
            if sym and i % 2 == 1:
                poles[:, i] = 0
            else:
                poles[:, i] = simps(xi * v[:, i][None, :], mu) / norm

        return poles

    @property
    def monopole(self):
        return self.poles[:, 0]
    @property
    def dipole(self):
        return self.poles[:, 1]
    @property
    def quadrupole(self):
        return self.poles[:, 2]

    def extract(self, rmask, mumask):
        """ extract a portion of the CorrFunc """
        return CorrFunc(self.r[rmask], self.mu[mumask],
                self.xi[rmask][:, mumask])

    def copy(self):
        return CorrFunc(self.r, self.mu, self.xi.copy())

    @staticmethod
    def QQ(r, mu, DQDQ, RQDQ, RQRQ, ratio):
        mu = mu[len(mu)//2:]
        xifull = (musym(DQDQ) \
                - musym(RQDQ) * 2 * ratio) \
                / (musym(RQRQ) * ratio ** 2) + 1
        xi = xifull[1:-1, 0:-1]
        return CorrFunc(r, mu, xi)
    @staticmethod
    def QF(r, mu, DQDFsum1, RQDFsum1, RQDFsum2, ratio):
        xifull = (DQDFsum1 - RQDFsum1 * ratio) \
                / (RQDFsum2 * ratio)
        xi = xifull[1:-1, 1:-1]
        return CorrFunc(r, mu, xi)

    @staticmethod
    def FF(r, mu, DFDFsum1, DFDFsum2):
        xifull = musym(DFDFsum1) / musym(DFDFsum2)
        xi = xifull[1:-1, 0:-1]
        mu = mu[len(mu)//2:]
        return CorrFunc(r, mu, xi)

    def __repr__(self):
        a = numpy.get_printoptions()
        try:
            numpy.set_printoptions(threshold=4, edgeitems=2)
            return "<CorrFunc (%d, %d) on r=%s, mu=%s>" % \
                    (self.xi.shape[0], self.xi.shape[1], str(self.r), str(self.mu))
        finally:
            numpy.set_printoptions(**a)


    def __getstate__(self):
        return (self.r, self.mu, self.xi)

    def __setstate__(self, state):
        self.r, self.mu, self.xi = state

class CorrFuncCollection(list):
    def __init__(self, funcs):
        list.__init__(self, funcs)

    def copy(self):
        return CorrFuncCollection([f.copy() for f in self])

    @Lazy
    def imesh(self):
        return numpy.repeat(numpy.arange(len(self)),
                [func.xi.size for func in self])

    @Lazy
    def rmesh(self):
        return numpy.concatenate(
                [func.rmesh.flat
                    for func in self])
    @Lazy
    def mumesh(self):
        return numpy.concatenate(
                [func.mumesh.flat
                    for func in self])

    def extract(self, mask):
        """ extract based on a global mask of r, mu mesh 
            NOTE: the mask has to be seperatable for different
            CorrFunc components.

            easiest way to guarentee this is to use
            (.rmesh > ???) & (.mumesh > ???)
        """
        offset = 0
        newfuncs = []
        for func in self:
            mymask = mask[offset:offset+func.xi.size]
            mymask = mymask.reshape(func.xi.shape)
            # now lets use the assumptiong that mymask is seperatable
            rmask = mymask.any(axis=1)
            mumask = mymask.any(axis=0)
            newfuncs.append(func.extract(rmask, mumask))
            offset += func.xi.size
        
        return CorrFuncCollection(newfuncs)

    def compress(self):
        """ return an array including all dofs
            of the correlation functions inside the collection
        """
        return numpy.concatenate(
                [func.xi.flat
                    for func in self])

    def uncompress(self, xicompressed):
        """fill the correlation function with compressed xi compress to self,
           returns self for chaining """
        offset = 0
        assert len(xicompressed) == numpy.sum([func.xi.size for func in self])
        for func in self:
            func.xi.flat[...] = xicompressed[offset:]
            if hasattr(func, 'poles'):
                del func.poles
            offset += func.xi.size
        return self

def musym(arr):
    """ symmetrize the mu (last) direction of a (r, mu) matrix,
        the size along mu direction will be halved! 
        **we do not divided by 2 **
        assume mu goes from -1 to 1 (ish)
        """
    N = arr.shape[-1]
    assert N % 2 == 0
    h = N // 2
    res = arr[..., h-1::-1] + arr[..., h:]
    return res


if __name__ == '__main__':
    from sys import argv
    config = Config(argv[1])
