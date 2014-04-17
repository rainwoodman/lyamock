import numpy
from kdcount import correlate
import sharedmem
import os.path
from common import Config
from sys import stdout

from qsocorr import getqso, getrandom
from common import CorrFunc
def main(A):
    data = correlate.points(getqso(A))
    random = correlate.points(getrandom(A))
    binning = correlate.RmuBinning(160000, Nbins=40, Nmubins=48, observer=0)
    DD = correlate.paircount(data, data, binning)
    DR = correlate.paircount(data, random, binning)
    RR = correlate.paircount(random, random, binning)
    r = 1.0 * len(data) / len(random)
    xi = (DD.sum1 + r ** 2 * RR.sum1 - 2 * r * DR.sum1) / (r ** 2 * RR.sum1)
    func = CorrFunc(DD.centers[0], DD.centers[1], xi)
    numpy.savez(os.path.join(A.datadir, 'qsocorr-Rmu.npz'), 
        center=DD.centers, xi=xi, corr=func)

if __name__ == '__main__':
    from sys import argv
    main(Config(argv[1]))
