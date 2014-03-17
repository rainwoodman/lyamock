import numpy
from scipy.integrate import cumtrapz

def splat0(t, value, bins):
    """put value into bins according to t
       the points are assumed to be describing a continuum field,
       if two points have the same position, they are merged into one point

       for points crossing the edge part is added to the left bin
       and part is added to the right bin.
       the sum is conserved.
    """
    if len(t) == 0:
        return numpy.zeros(len(bins) + 1)
    t = numpy.array(t, copy=True, dtype='f8')
    t, label = numpy.unique(t, return_inverse=True)
    if numpy.isscalar(value):
        value = numpy.bincount(label) * value
    else:
        value = numpy.bincount(label, weights=value)
    edge = numpy.concatenate(([t[0]], (t[1:] + t[:-1]) * 0.5, [t[-1]]))
    dig = numpy.digitize(edge, bins)
    #use the right edge as the reference
    ref = bins[dig[1:] - 1]
    norm = (edge[1:] - edge[:-1])
    assert ((edge[1:] - edge[:-1]) > 0).all()
    norm = 1 / norm
    weightleft = -(edge[:-1] - ref) * norm
    weightright = (edge[1:] - ref) * norm
    # when dig < 1 or dig >= len(bins), t are out of bounds and does not
    # contribute.
    l = numpy.bincount(dig[:-1], value * weightleft, minlength=len(bins)+1)
    r = numpy.bincount(dig[1:], value * weightright, minlength=len(bins)+1)
    return l + r

def splat(t, value, bins):
    """put value into bins according to t
       the points are assumed to be describing a continuum field,
       if two points have the same position, they are merged into one point

       for points crossing the edge part is added to the left bin
       and part is added to the right bin.
       the sum is conserved.
    """
    if len(t) == 0:
        return numpy.zeros(len(bins) + 1)
    ind = numpy.argsort(t)
    t = t[ind]
    value = value[ind]
    cum = numpy.concatenate(([0], numpy.cumsum(value)))
    newcum = numpy.interp(bins, t, cum[:-1])
    return numpy.concatenate(([newcum[0]], 
                    numpy.diff(newcum),
                    [cum[-1] - newcum[-1]]))

