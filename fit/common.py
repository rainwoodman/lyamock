import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from lib.common import *

def writepickle():
    import pickle
    from sys import stdout
    mu = numpy.linspace(-1, 1, 10)
    r = numpy.linspace(-1, 1, 10)
    xi = numpy.empty((len(r), len(mu)))
    s = pickle.dumps(CorrFuncCollection([CorrFunc(r, mu, xi)]))
    stdout.write(s)

def readpickle():
    import pickle
    from sys import stdin
    print pickle.load(stdin)

