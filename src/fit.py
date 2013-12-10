import sharedmem
import numpy
from cosmology import interp1d
from scipy.special import sph_jn, legendre
from scipy.integrate import quad
from scipy.optimize import fmin, minimize
from kdcount import correlate
from scipy import linalg
import os.path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("power_spectrum", help="txt file, from main.py")
    parser.add_argument("covarience_matrix", help="npz file, cov.py")
    parser.add_argument("mock", help="corrbootstrap.npz bootstrap/main.py")
    parser.add_argument("rmin", help="rmin", type=float)
    parser.add_argument("rmax", help="rmax", type=float)
    parser.add_argument("outputf", help="FF analysis output filename (npz)")
    parser.add_argument("outputj", help="joined analysis filename (npz)")

    # this is hard coded, should have been saved by cov.py as well as in the
    # bootstraps
    binning = correlate.RmuBinning(80000, Nbins=20, Nmubins=48, observer=0)
    r, mu = binning.centers
    A = parser.parse_args()
    rmin = A.rmin
    rmax = A.rmax
    print rmin, rmax
    rselection = (r >= rmin) & (r <= rmax)
    print rselection
    # watch out for the edge bins
    chunks = numpy.load(os.path.join(A.mock))
    k, p = numpy.loadtxt(A.power_spectrum, unpack=True)
    cov = numpy.load(A.covarience_matrix)['cov']


    dataFF = chunks['DFDFsum1'].sum(axis=-1)[1, 1:-1, 1:-1] \
            / chunks['DFDFsum2'].sum(axis=-1)[1:-1, 1:-1]
    # apply r selection
    print dataFF.shape
    ratio = 1.0 * chunks['Qchunksize'].sum() / chunks['Rchunksize'].sum()
    dataQF = (chunks['DQDFsum1'].sum(axis=-1)[1, 1:-1, 1:-1]  -
             ratio * chunks['RQDFsum1'].sum(axis=-1)[1, 1:-1, 1:-1]) \
            / (ratio * chunks['RQDFsum2'].sum(axis=-1)[1:-1, 1:-1])
    # apply r selection
    dataFF = dataFF[rselection, :]
    dataQF = dataQF[rselection, :]

    cov_FF = cov[2, :, :, 2, :, :][rselection, ...][..., rselection, :]
    mf = ModelFactoryFF(k, p, r[rselection], mu)
    FF = FFanalysis(mf, dataFF, cov_FF)

    numpy.savez(A.outputf, **FF)

    # QF is 1 in cov
    dataJoined = numpy.array([dataQF, dataFF])
    cov_Joined = cov[1:3, :, :, 1:3, :, :][:, rselection, ...][..., rselection, :]
    mf = ModelFactoryJoined(k, p, r[rselection], mu)
    Joined = Joinedanalysis(mf, dataJoined, cov_Joined)

    numpy.savez(A.outputj, **Joined)
    

def FFanalysis(mf, data, cov):
    r = mf.r
    mu = mf.mu

    # let's do FF for now
    # apply r selection to cov
    invcov = covinv(cov)
    LinvcovL = numpy.einsum('lm,rmsn,un->lrus', mf.L, invcov, mf.L)
    cov2 = covinv(LinvcovL)
    err = covdiag(cov2) ** 0.5

    def costfunc(b, B):
        model = mf(C0FF(b, B), C2FF(b, B), C4FF(b, B))
        return numpy.einsum('ij,ijkl,kl', data - model, invcov, data - model)

    res = minimize(lambda x: costfunc(*x), (-0.2, 1.5))
    print res.x
    b, B = res.x
    nddof = data.size - 2
    chi = costfunc(b, B) / nddof
    print 'xisq per dof', chi, 'ddof', nddof
    bgrid = numpy.linspace(-0.5, -0.1, 40)
    Bgrid = numpy.linspace(0.5, 2.0, 80)
    bmesh,Bmesh = numpy.meshgrid(bgrid, Bgrid, indexing='ij')
    cost = numpy.vectorize(costfunc)(bmesh, Bmesh) / nddof - chi
    model = mf(C0FF(b, B), C2FF(b, B), C4FF(b, B))
            
    return dict(cost=cost, b=bgrid, B=Bgrid, b0=b, B0=B,
            model=model, data=data, r=r, mu=mu, err=err, chi=chi,
            ddof=nddof)

def JoinedAnalysis(mf, data, cov):
    r = mf.r
    mu = mf.mu

    # let's do FF for now
    # apply r selection to cov
    invcov = covinv(cov)
    LinvcovL = numpy.einsum('lm,armbsn,un->alrbus', mf.L, invcov, mf.L)
    cov2 = covinv(LinvcovL)
    err = covdiag(cov2) ** 0.5

    def modelfunc(bF, BF, bQ, BQ):
        model = mf(C0FF(bF, BF), C2FF(bF, BF), C4FF(bF, BF), C0QF(bQ, BQ, bF,
            BF), C2QF(bQ, BQ, bF, BF))
        return model
    def costfunc(bF, BF, bQ, BQ):
        model = modelfunc(bF, BF, bQ, BQ)
        return numpy.einsum('aij,aijbkl,bkl', data - model, invcov, data - model)

    res = minimize(lambda x: costfunc(*x), (-0.2, 1.4, 2.5, 1.5))
    print res.x
    bF, BF, bQ, BQ = res.x
    nddof = data.size - 4
    chi = costfunc(bF, BF, bQ, BQ) / nddof
    print 'xisq per dof', chi, 'ddof', nddof
    model = modelfunc(bF, BF, bQ, BQ)
            
    return dict(bF0=bF, BF0=BF, bQ0=bQ, BQ0=BQ,
            model=model, data=data, r=r, mu=mu, err=err, chi=chi,
            ddof=nddof)

def scalarize(func):
    def wrapped(*args, **kwargs):
        if numpy.isscalar(args[0]):
            args = [numpy.array([args[0]])] + list(args[1:])
            rt = func(*args, **kwargs)
            return rt[0]
        else:
            return func(*args, **kwargs)
    return wrapped
@scalarize
def j0(x):
    out = numpy.empty_like(x)
    out = numpy.sin(x) / x
    out[numpy.isnan(out)] = 1.0
    return out   

@scalarize
def j2(x): # this is negative of 4.11 in 1104.5244v2.pdf
    s = numpy.sin(x)
    c = numpy.cos(x)
    out = -(s * x ** 2 - 3 * s + 3 * c * x) / x ** 3
    out[numpy.isnan(out)] = 0.0
    return out   

@scalarize
def j4(x):
    out = numpy.empty_like(x)
    s = numpy.sin(x)
    c = numpy.cos(x)
    out = (x ** 4 *  s - 45 * x ** 2* s + 105 * s + 10 * x ** 3 * c - 105 * c * x) / x ** 5
    out[numpy.isnan(out)] = 0.0
    return out   

def C0FF(b, B):
    return b ** 2 * (1 + 2. / 3. * B + 1. / 5. * B ** 2)
def C2FF(b, B):
    return b ** 2 * (4. / 3. * B + 4. / 7. * B ** 2)
def C4FF(b, B):
    return b ** 2 * (8. / 35. * B ** 2)

def C0QF(bQ, BQ, bF, BF):
   return bF * bQ * (1 + 1./3 * (BF + BQ) + 1. / 5 * BF * BQ)

def C2QF(bQ, BQ, bF, BF):
   return bF * bQ *(2./3 * (BF + BQ) + 4. / 7 * BF * BQ)

class ModelFactoryBase(object):
    def __init__(self, k, p, r, mu):
        """ factory to make a model for the powerspectrum given with k, p.
            on a r , mu grid
            k in kpc/h units p in gadget convention. (k, p decided by Omega
            stuff)

            xi0, xi2, and xi4 are calculated with scipy.quad; 
               looks impressive and it's fast.

            rumor is k, p from camb is not good (Ross/ ask Xiaoying). 
       """
        self.xi0 = powertopole(r, k, p, kernel=j0)
        self.xi2 = powertopole(r, k, p, kernel=j2)
        self.xi4 = powertopole(r, k, p, kernel=j4)
        self.r = r
        self.mu = mu
        P0 = legendre(0)(mu)
        P2 = legendre(2)(mu)
        P4 = legendre(4)(mu)
        self.L = numpy.empty((3, len(mu)))
        self.L[0] = P0
        self.L[1] = P2
        self.L[2] = P4


class ModelFactoryFF(ModelFactoryBase):
    def __init__(self, k, p, r, mu):
        ModelFactoryBase.__init__(self, k, p, r, mu)

    def __call__(self, C0FF, C2FF, C4FF):
        """returns a model xi evalucated at r, mu grid, 
        
           inputs:
               the tracer bias b,
               and the RSD factors C0, C2, C4 for the tracers 

           For forest (see Anze:1104.5244v2)

               C0 = b**2 * (1 + 2./3 *B + 1./5 * B ** 2)
               C2 = b **2 *(4./3 * B + 4./7 * B ** 2)
               C4 = b **2 * (8. / 35 * B ** 2)
       """
        #print 'Cfactors', C0, C2, C4
        P_l = self.L
        xi_Fl = [self.xi0, self.xi2, self.xi4]
        C_l = [C0FF, C2FF, C4FF]
        # our j2 is negative of 4.11
        sign = [1, -1, 1,]
        xi_F = numpy.sum([
              xi_Fl[i][:, None] * P_l[i][None, :] 
              * C_l[i] * sign[i]
              for i in range(3)], axis=0)
        return xi_F

class ModelFactoryJoined(ModelFactoryBase):
    def __init__(self, k, p, r, mu):
        ModelFactoryBase.__init__(self, k, p, r, mu)

    def __call__(self, C0FF, C2FF, C4FF, C0QF, C2QF):
        """returns a model xi evalucated at r, mu grid, 
           shape = 2 , len(r) , len(mu)
       """
        #print 'Cfactors', C0, C2, C4
        xi_F = ModelFactoryFF.__call__(self, C0FF, C2FF, C4FF)

        P_l = self.L
        xi_Fl = [self.xi0, self.xi2, self.xi4]
        # our j2 is negative of 4.11
        sign = [1, -1, 1,]
        C_l = [C0QF, C2QF]
        xi_QF = numpy.sum([
              xi_Fl[i][:, None] * P_l[i][None, :] 
              * C_l[i] * sign[i]
              for i in range(2)], axis=0)
        # watch out QF is stored before FF in the full cov matrix
        return numpy.array([xi_QF, xi_F])


def powertopole(r, K, P, kernel):
    """calculate the multipoles of xi upto order nmax, for given P(K)
       K R are in DH units!
    """
    mask = ~numpy.isnan(P) & (K > 0)
    K = K[mask]
    P = P[mask]
    P = P * (2 * numpy.pi) ** 3 # going from GADGET to xiao
    Pfunc = interp1d(K, P, kind=5)

    xi = numpy.empty_like(r)
    for i in range(len(r)):
        def func(k):
            rt = Pfunc(k) * k ** 2 * numpy.exp(- (k *1e3) ** 2) * kernel(k * r[i])
            return rt
        xi[i] = quad(func, K.min(), K.max())[0]
    return xi * (2 * numpy.pi) ** -3
def covinv(cov):
    """ invert cov
    returns inversion of cov in the same shape as original
    assume cov has shape
    (a, b, c, ..., a, b, c, ...)
    """
    flat = covflat(cov)
    inv = linalg.inv(flat)
    return inv.reshape(cov.shape)

def covflat(cov):
    """ flattens a cov matrix (flatten the dof), making
       cov matrix a square matrix
    """
    assert len(cov.shape) % 2 == 0
    Nd = len(cov.shape) / 2
    len1 = numpy.prod(cov.shape[:Nd])
    len2 = numpy.prod(cov.shape[Nd:])
    assert len1 == len2
    return cov.reshape(len1, len2)

def covdiag(cov):
    """ extract the diagoanl terms in cov in the dof shape
        ** 0.5 gives the err estimate
    """
    flat = covflat(cov)
    diag = numpy.diag(flat)
    Nd = len(cov.shape) / 2
    return diag.reshape(cov.shape[:Nd])


if __name__ == "__main__":
    main()
