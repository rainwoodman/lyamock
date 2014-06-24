import sys
sys.path.append('../')

from fit.common import *
from fit.bootstrap import MakeEigenModes
from sys import argv

def fixdir(dir):
    config = Config(dir + '/paramfile', dir)
    f = numpy.load(dir + '/datadir/bootstrap.npz')
    dict = {}
    for f in f.files:
        dict[f] = f[f]

    powerspec = PowerSpectrum(config)
    eigenvalues = fit.
