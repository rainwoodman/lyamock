import numpy
import sharedmem
from common import CovConfig
from common import PowerSpectrum
from common import MakeEmptySample
from common import EigenModes
from lib.chunkmap import chunkmap
from numpy.polynomial.legendre import legval
from numpy import linalg
from bootstrap import RmuBinningIgnoreSelf

def main(config):
    """ this code generates the fitting eigen states 
        because it will be reused  
    """
    binning = RmuBinningIgnoreSelf(160000, Nbins=40, Nmubins=48, 
            observer=0)
    r, mu = binning.centers
    powerspec = PowerSpectrum(config, cutscale=8000)
    dummy = MakeEmptySample(r, mu)
    eigenmodes = MakeEigenModes(powerspec, dummy)
    
    numpy.savez(config.EigenModesOutput, eigenmodes=eigenmodes)

def MakeEigenModes(powerspec, templatecorrfunc):
    """ create fitting eigenstates for a given powerspectrum
        see Slosar paper.

        basically getting out the poles of the powerspectrum,
        then evaluate them on the grid given in templatecorrfunc.
        this is used in bootstrap (for per sample)

        usually, the eigenmodes are ordered first by the correlation functions, 
        (QQ QF, FF) then by the multipole order (0, 2, 4)
    """
    dummy = templatecorrfunc

    eigenmodes = []
    N = len(dummy.compress())
    annotation = []

    for i in range(len(dummy)):
        for order in [0, 2, 4]:
            annotation.append((i, order))
            eigenmode = dummy.copy()
            eigenmode.uncompress(numpy.zeros(N))
            eigenmode[i].xi = sharedmem.copy(eigenmode[i].xi)
            eigenmodes.append(eigenmode)

    with sharedmem.MapReduce() as pool:
        def work(j):
            i, order = annotation[j]
            c = numpy.zeros(5)
            c[order] = 1
            # watch out the eigenmode is wikipedia
            # no negative phase on 2nd order pole!
            eigenmode = eigenmodes[j]
            eigenmode[i].xi[...] = \
                    powerspec.pole(eigenmode[i].r, order=order)[:, None] \
                    * legval(eigenmode[i].mu, c)[None, :]
        pool.map(work, range(len(annotation)))
    # the eigenmodes are ordered first by the correlation functions, (QQ QF, FF)
    # then by the multipole order (0, 2, 4)
    #
    for j in  range(len(annotation)):
        i, order =annotation[j]
        eigenmode = eigenmodes[j]
        eigenmode[i].xi = numpy.array(eigenmode[i].xi, copy=True)

    return EigenModes(eigenmodes)


if __name__ == '__main__':
    from sys import argv
    main(CovConfig(argv[1]))
