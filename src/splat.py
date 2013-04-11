import numpy
def splat(t, value, bins):
    """put value into bins according to t
       the points are assumed to be describing a continuum field,
       thus no two points have the same position.
       they are also supposed to be sorted.

       for points crossing the edge part is added to the left bin
       and part is added to the right bin.
       the sum is conserved.
    """
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

