import numpy
import os.path
import StringIO
import ConfigParser
import argparse
from scipy.interpolate import interp1d

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
  args.Sigma8 = config.getfloat("Cosmology", "Sigma8")
  args.OmegaM = config.getfloat("Cosmology", "OmegaM")
  args.OmegaB = config.getfloat("Cosmology", "OmegaB")
  args.OmegaL = config.getfloat("Cosmology", "OmegaL")
  args.h = config.getfloat("Cosmology", "h")
  args.G = 43007.1
  args.H = 0.1

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
    import pycamb
    a = dict(H0=args.h * 100, 
          omegac=args.OmegaM - args.OmegaB, 
          omegab=args.OmegaB, 
          omegav=args.OmegaL, 
          omegak=0, omegan=0)
    fakesigma8 = pycamb.transfers(scalar_amp=1, **a)[2]
    scalar_amp = fakesigma8 ** -2 * args.Sigma8 ** 2
    k, p = pycamb.matter_power(scalar_amp=scalar_amp, maxk=10, **a)
    k /= 1000
    p *= 1e9 * (2 * numpy.pi) ** -3 # xiao to GADGET
    p[numpy.isnan(p)] = 0
    return interp1d(k, p, kind='linear', copy=True, bounds_error=False, fill_value=0)

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
