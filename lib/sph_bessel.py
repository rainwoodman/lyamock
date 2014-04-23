import numpy
# these functions are from
# http://xaldev.sourceforge.net/javadoc/xal/tools/math/BesselFunction.html#j4%28double%29


def scalarize(func):
    def wrapped(*args, **kwargs):
        if numpy.isscalar(args[0]):
            args = [numpy.array([args[0]])] + list(args[1:])
            rt = func(*args, **kwargs)
            return rt[0]
        else:
            return func(*args, **kwargs)
    wrapped.__doc__ = func.__doc__
    return wrapped

@scalarize
def j0(x, out=None):
    if out is None:
        out = numpy.empty_like(x)
    mask = x > 1e-4
    x = x[mask]
    out[mask] = numpy.sin(x) / x
    out[~mask] = 1.0
    return out   

@scalarize
def j2(x, out=None): 
    # this is negative of 4.11 in 1104.5244v2.pdf
    # same as wikipedia
    if out is None:
        out = numpy.empty_like(x)
    mask = x > 1e-3
    xs = x[~mask]
    x = x[mask]
    s = numpy.sin(x)
    c = numpy.cos(x)
    out[mask]= -(s * x ** 2 - 3 * s + 3 * c * x) / x ** 3
    out[~mask] = xs**2 / 15 - xs**4 / 210. + xs**6  /7560 - xs**8/498960
    return out   

@scalarize
def j4(x, out=None):
    if out is None:
        out = numpy.empty_like(x)
    mask = x >1e-3
    xs = x[~mask]
    x = x[mask]

    s = numpy.sin(x)
    c = numpy.cos(x)
    out[mask] = (1 - 45 / x**2 + 105 / x**4) * s / x \
            + (10 / x - 105 / x**3) * c / x
    out[~mask] = xs**4 / 945 - xs**6 / 20790 + xs**8 / 1081080
    return out   

