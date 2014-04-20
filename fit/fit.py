import numpy
import sharedmem
from common import CovConfig, BootstrapDB, EigenModes
from common import PowerSpectrum
from common import CorrFuncCollection
from bootstrap import MakeBootstrapSample
from lib.chunkmap import chunkmap
from scipy.optimize import minimize
from numpy.polynomial.legendre import legval, legvander
from numpy import linalg

def main(config):
    global cov
    DB = BootstrapDB(config)

    MASK = (DB.dummy.imesh >= 0)
    MASK &= (DB.dummy.rmesh <= config.rmax)
    MASK &= (DB.dummy.rmesh >= config.rmin)

    print 'dof in fitting', MASK.sum()
    # create a dummy to test the fitting
    p0 = [-0.2, 3.5, 1.5, 1.5]

    eigenmodes = DB.eigenmodes
    dummy = eigenmodes(p0)

    covfull = numpy.load(config.CovarianceMatrixOutput)['cov']
    cov = covfull[MASK][:, MASK]
    
    print 'inverting'
    INV = linalg.inv(covfull[MASK][:, MASK])
    print 'inverted'

    x, chi = fit1(dummy, eigenmodes, INV, MASK)

    print 'x =', x
    print 'p0 =', p0

    error = poles_err(dummy, covfull)

    fitted = sharedmem.empty((len(DB), len(p0)))
    chi = sharedmem.empty((len(DB)))
    samples, models = [], []
    sharedmem.set_debug(True)
    def work(i):
        sample = DB(i)
        print 'fitting', i
        fitted[i], chi[i] = fit1(sample, eigenmodes, INV, MASK)
        model = eigenmodes(fitted[i])
        print zip(sample[0].monopole, model[0].monopole)
        return sample, model
    def reduce(rt):
        s, m = rt
        samples.append(s)
        models.append(m)
    chunkmap(work, range(len(DB)), 100, reduce=reduce)
    numpy.savez("fit.npz", samples=samples, models=models, 
            fittedparameters=fitted, chi=chi,
            error=error)

def covtopole(dummy, cov):
   # cov[numpy.isnan(cov)] = 0.0 
    newdof = numpy.sum([func.poles.size for func in dummy])
    tmp = numpy.empty((newdof, cov.shape[0]))
    for i in range(cov.shape[0]):
        tmp[:, i] = numpy.concatenate(
            [
            func.poles.flat
            for func in dummy.uncompress(cov[:, i])
            ])
     
    newcov = numpy.empty((newdof, newdof))
    for i in range(tmp.shape[0]):
        newcov[i, :] = numpy.concatenate(
            [
            func.poles.copy().flat
            for func in dummy.uncompress(tmp[i, :])
            ])
    return newcov

def poles_err(dummy, cov):
    newcov = covtopole(dummy, cov)
    rt = dummy.copy()
    err = numpy.diag(newcov) ** 0.5
    offset = 0
    for func in rt:
        func.frompoles(err[offset:offset+func.poles.size].copy().reshape(len(func.r), -1))
        offset += func.poles.size
    return rt

def fit1(sample, eigenmodes, INV, mask):
    p0 = [-0.4, 2, 2.5, 2.5]
    # we only want the part that is used in the covmatrix
    xi = sample.compress()[mask]
    def cost(p):
        model = eigenmodes(p).compress()[mask]
        diff = model - xi
        fac = 1.0 / (model.size - len(p))
        cost = numpy.einsum('i,ij,j', diff, INV, diff) * fac
        assert cost >= 0.0
        print cost, p
        return cost
    res = minimize(cost, p0, tol=1e-4) 
    print res.success, res.x, cost(res.x) ** 0.5
    return res.x, cost(res.x) ** 0.5

if __name__ == '__main__':
    from sys import argv
    main(CovConfig(argv[1]))
