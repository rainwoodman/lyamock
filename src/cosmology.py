import numpy
from scipy.integrate import romberg
from scipy.interpolate import InterpolatedUnivariateSpline

class Lazy(object):
    def __init__(self, calculate_function):
        self._calculate = calculate_function

    def __get__(self, obj, _=None):
        if obj is None:
            return self
        value = self._calculate(obj)
        setattr(obj, self._calculate.func_name, value)
        return value

class interp1d(InterpolatedUnivariateSpline):
  """ this replaces the scipy interp1d which do not always
      pass through the points
      note that kind has to be an integer as it is actually
      a UnivariateSpline.
  """
  def __init__(self, x, y, kind, bounds_error=False, fill_value=numpy.nan, copy=True):
    if copy:
      self.x = x.copy()
      self.y = y.copy()
    else:
      self.x = x
      self.y = y
    InterpolatedUnivariateSpline.__init__(self, self.x, self.y, k=kind)
    self.xmin = self.x[0]
    self.xmax = self.x[-1]
    self.fill_value = fill_value
    self.bounds_error = bounds_error
  def __call__(self, x, nu=0):
    x = numpy.asarray(x)
    shape = x.shape
    x = x.ravel()
    bad = (x > self.xmax) | (x < self.xmin)
    if self.bounds_error and numpy.any(bad):
      raise ValueError("some values are out of bounds")
    y = InterpolatedUnivariateSpline.__call__(self, x.ravel(), nu=nu)
    y[bad] = self.fill_value
    return y.reshape(shape)
 
Np = 1000
logamin = -20
class Cosmology(object):
  """ every thing is dimensionless.
      Length is in c / H0
  """

  def __init__(self, M, L, h, B, sigma8):
    self.M, self.L = M, L
    self.h = h
    self.sigma8 = sigma8
    self.B = B
  @Lazy
  def Dc(self):
    M, L = self.M, self.L
    logx = numpy.linspace(logamin, 0, Np)
    def kernel(loga):
      a = numpy.exp(loga)
      return 1 / self.Ea(a) * a ** -1 # dz = - 1 / a dloga
    y = numpy.array(
        [romberg(kernel, loga, 0, vec_func=True, divmax=10) for loga in logx])
    def func(xval, 
            intp=interp1d(logx, y, kind=5)):
      return intp(numpy.log(xval))
    func.y = y
    func.x = numpy.exp(logx)
    func.__doc__ =  \
    """evaluates Dc(a) / DH
       (dimensionless, not multiplied by H0)
       for OmegaM = %g, OmegaL = %g
    """ % (M, L)
    return func

  @Lazy
  def aback(self):
    M, L = self.M, self.L
    y = self.Dc.x[::-1]
    x = self.Dc.y[::-1]
    assert (x[1:] > x[:-1]).all()
    def func(xval, nu=0,
            intp=interp1d(x, y, 
               bounds_error=True, 
               fill_value=numpy.nan, kind=5)):
      return intp(xval, nu=nu)
    func.__doc__ = \
    """evaluates look-back time 
       (in terms of expansion factor a)
       at given dimensionless comoving distance
       (Dc/DH)
    """
    return func

  @Lazy
  def Dplus(self):
    M, L = self.M, self.L
    logx = numpy.linspace(logamin, 0, Np)
    x = numpy.exp(logx)
  
    def kernel(loga):
      a = numpy.exp(loga)
      return (a * self.Ea(a)) ** -3 * a # da = a * d loga
  
    y = self.Ea(x) * numpy.array(
                 [ romberg(kernel, logx.min(), loga, vec_func=True) 
                   for loga in logx])

    def func(x, nu=0, intp=interp1d(logx, y, 
            bounds_error=False, 
            fill_value=numpy.nan, kind=5)):
      return intp(numpy.log(x), nu=nu)
    func.__doc__ =  \
    """evaluates D+(a) There is no 2.5 factor neither.
       (dimensionless, not multiplied by H0)
       for OmegaM = %g, OmegaL = %g
    """ % (M, L)
    return func

  @Lazy
  def Ea(self):
    M, L = self.M, self.L
    func = lambda a: \
         a ** -1.5 * (M + (1 - M - L) * a + L * a ** 3) ** 0.5
    func.__doc__ = \
    """evaluates H(a)/H0 (dimensionless) 
       for OmegaM = %g, OmegaL = %g
    """ % (M, L)
    return func 

  @Lazy
  def FOmega(self):
    M, L = self.M, self.L
    def func(a):
      omega_a = M / (M + (1 - M - L) * a + L * a ** 3)
      return omega_a ** 0.6
      
    func.__doc__ = \
    """evaluates FOmega(a) 
       for OmegaM = %g, OmegaL = %g
    """ % (M, L)
    return func 

  @Lazy
  def Pk(self):
    import pycamb
    a = dict(H0=self.h * 100., 
          omegac=self.M- self.B, 
          omegab=self.B, 
          omegav=self.L, 
          omegak=1 - self.M - self.L, omegan=0)
    DH = 299792.468 / 100.
    fakesigma8 = pycamb.transfers(scalar_amp=1, **a)[2]
    scalar_amp = fakesigma8 ** -2 * self.sigma8 ** 2
    k, p = pycamb.matter_power(scalar_amp=scalar_amp, maxk=500, **a)
    k *= DH
    p /= DH ** 3
    p[numpy.isnan(p)] = 0
    func = interp1d(k, p, kind=5, bounds_error=False, fill_value=0)
    func.__doc__ = \
    """power spectrum P(k) normalized to sigma8.
       k is in 1/DH and p is in DH**3
    """
    return func

WMAP7 = Cosmology(M=0.272, L=0.728, h=0.702, B=0.046, sigma8=0.801)
