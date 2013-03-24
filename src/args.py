import numpy
import os.path
import StringIO
import ConfigParser
import argparse
from scipy.interpolate import interp1d
from cosmology import Cosmology

pixeldtype = numpy.dtype(
    [ ('pos', ('f4', 3)), ('delta', 'f4'), ('losdisp', 'f4'), ('row', 'i4'), ('ind', 'i4'), ('Z', 'f4')])

def parseargs(argv=None):
  parser = argparse.ArgumentParser(description="")
  parser.add_argument("paramfile", 
               help="the paramfile")
  parser.add_argument("i",
               help="which row to do", type=int)
  args = parser.parse_args(argv)
  config = ConfigParser.ConfigParser()
  str = file(args.paramfile).read().replace(';', ',').replace('#', ';')
  config.readfp(StringIO.StringIO(str))

  args.datadir = config.get("IO", "datadir")
  args.BoxSize = config.getfloat("IC", "BoxSize")
  args.Redshift = config.getfloat("IC", "Redshift")
  args.Nmesh = config.getint("IC", "Nmesh")
  args.NmeshEff = config.getint("IC", "NmeshEff")

  args.Nrep = args.NmeshEff // args.Nmesh
  assert args.NmeshEff % args.Nmesh  == 0

  args.Seed = config.getint("IC", "Seed")
  Sigma8 = config.getfloat("Cosmology", "Sigma8")
  OmegaM = config.getfloat("Cosmology", "OmegaM")
  OmegaB = config.getfloat("Cosmology", "OmegaB")
  OmegaL = config.getfloat("Cosmology", "OmegaL")
  h = config.getfloat("Cosmology", "h")
  args.G = 43007.1
  args.C = 299792.458
  args.H = 0.1
  args.cosmology = Cosmology(M=OmegaM, 
            L=OmegaL, B=OmegaB, h=h, sigma8=Sigma8)
  args.Offset = numpy.array(numpy.unravel_index(numpy.arange(args.Nrep ** 3),
             (args.Nrep, ) * 3)).T.reshape(args.Nrep, args.Nrep, args.Nrep, 3) \
               * args.BoxSize / args.Nrep
  print args.Nrep
  numpy.random.seed(args.Seed)
  args.seed0 = numpy.random.randint(1<<21 - 1)
  args.seedtable = numpy.random.randint(1<<21 - 1, size=(args.Nrep,) * 3)
  try:
    power = config.get("IC", "PowerSpectrum")
    k, p = numpy.loadtxt(power, unpack=True)
    p[numpy.isnan(p)] = 0
    args.power = interp1d(k, p, kind='linear', copy=True, 
                    bounds_error=False, fill_value=0)
    print 'using text file', power
  except ConfigParser.NoOptionError:
    print 'using PyCAMB'
    args.power = loadpower(args)
  try:
    args.pixeldir = config.get("IO", 'pixeldir')
    print 'using pixels from ', args.pixeldir
  except ConfigParser.NoOptionError:
    args.Npix = config.getint("IC", 'Npix')
    print 'using regular pixel grid'
    args.pixeldir = None
  return args

def loadpower(args):
    Pk = args.cosmology.Pk
    def func(k, Pk=args.cosmology.Pk, DH=args.C/args.H):
       return Pk(k * DH) * DH ** 3 * (2 * numpy.pi) ** -3
    func.x = Pk.x / DH
    func.y = Pk.y * DH ** 3 * (2 * numpy.pi) ** -3
    return func

def loadpixel(args, i, j, k):
  if args.pixeldir is not None:
    try:
      return numpy.fromfile(args.pixeldir + "%02d-%02d-%02d-pixels.raw" % (i, j, k), dtype=pixeldtype)
    except IOError:
      return numpy.array([], dtype=pixeldtype)
  else:
    Npix = args.Npix / args.Nrep
    pos = args.Offset[i, j, k] + \
        numpy.array(numpy.unravel_index(numpy.arange(Npix ** 3),
             (Npix, ) * 3)).T * (args.BoxSize / args.Nrep / Npix)
    rt = numpy.empty(shape=len(pos), dtype=pixeldtype)
    rt['pos'] = pos
    rt['ind'] = numpy.arange(Npix ** 3)
    rt['row'] = (i * args.Nrep * args.Nrep + j * args.Nrep + k)
    return rt
