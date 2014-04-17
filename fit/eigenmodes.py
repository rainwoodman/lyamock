import numpy
import sharedmem
from common import CovConfig, BootstrapDB
from common import PowerSpectrum
from common import MakeBootstrapSample
from common import MakeEigenStates
from lib.chunkmap import chunkmap
from numpy.polynomial.legendre import legval
from numpy import linalg

def main(config):
    """ this code generates the fitting eigen states 
        because it will be reused  
    """
    DB = BootstrapDB(config)
    powerspec = PowerSpectrum(config)
    
    eigenmodes = MakeEigenStates(powerspec, DB.dummy)
    # the eigenmodes are ordered first by the correlation functions, (QQ QF, FF)
    # then by the multipole order (0, 2, 4)
    #
    numpy.savez(config.EigenModesOutput, annotation=annotation, eigenmodes=eigenmodes)



if __name__ == '__main__':
    from sys import argv
    main(CovConfig(argv[1]))
