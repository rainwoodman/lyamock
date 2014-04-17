import numpy

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
    mask = x != 0
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
    mask = x != 0
    x = x[mask]
    s = numpy.sin(x)
    c = numpy.cos(x)
    out[mask]= -(s * x ** 2 - 3 * s + 3 * c * x) / x ** 3
    out[~mask] = 0.0
    return out   

@scalarize
def j4(x, out=None):
    if out is None:
        out = numpy.empty_like(x)
    mask = x != 0
    x = x[mask]
    s = numpy.sin(x)
    c = numpy.cos(x)
    out[mask] = (x ** 4 *  s - 45 * x ** 2* s + 105 * s + 10 * x ** 3 * c - 105 * c * x) / x ** 5
    out[~mask] = 0.0
    return out   

