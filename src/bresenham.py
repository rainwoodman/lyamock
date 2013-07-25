import numpy
def idiv_trunc(x, y, updown):
    """ return x / y up or down
        if updown is 1, up
        -1, down
    """

    s2 = numpy.sign(y)
    x = s2 * x
    y = s2 * y
    rt = (x + numpy.where(updown == 1, y - 1, 0)) // y 
    #print 'idve_trun', x, y, updown, 1.0 * x / y, rt
    return rt

def clipline(x1, x2, min, max, return_full=True):
    """
    returns the number of pixels per line
    if return_full is True
      returns x, o, p
      where x is the pixel pos in (-1, ndim),
      o and p are the index of the line and the relative pixel(0 is x1 and max
      is x2)

    Algorithm:
        The line is (x1, x2, t, T are integers)
            x = x1 + t / T * (x2 - x1)
        where t is in [0, T], and T is abs(max(x2 - x1)), along
        the longest axis (the prime). The choice inspired by bresenham,
        as it covers the entire line even if  t is integer. 


        The clipping is
            x >= min and x < max

        Now write the inequalities:
            t * (x2 - x1) >= (min - x1) * T,
            t * (x2 - x1) < (max- x1) * T

        the second is, for it's all integer
            t * (x2 - x1) <= (max - x1) * T - 1

        T is always positive. x2 - x1 can be negative or positive.
        so we need to handel cases to decide range of t.
        
        when x2 - x1 != 0,
            t1 = -(min - x1) * T / (x2 - x1)
            t2 = ((max - x1) * T - 1) / (x2 - x1)
        else:
            t1 = T + 1 if x1 < min or x1 >= max (empty)
                 -1 othersize  (all)
            t2 = -1 if x1 < min or x1 >= max
                 T + 1 othersize
            
        for axis with x2 - x1 >= 0, 
             t is in [up(t1), down(t2)], for that axis;
             this includes the case x2 - x1 == 0.
        when x2 - x1 < 0,
             t is in [up(t2), down(t1)], for that axis

        each allowed t value corresponds to one bresenhem pixel.
        
    """
    x2 = numpy.intp(x2)
    x1 = numpy.intp(x1)
    Dx = x2 - x1
    T = numpy.abs(Dx).max(axis=-1)
    bad = Dx == 0
    #protect against 0
    Dx[bad] = 1
    # we want to round towards 0
    min = numpy.array(min, dtype='intp')
    max = numpy.array(max, dtype='intp')
    t1 = idiv_trunc((min - x1) * T[..., None], Dx, numpy.sign(Dx))
    t2 = idiv_trunc((max - x1) * T[..., None] - 1, Dx, numpy.sign(Dx) * -1)
    Dx[bad] = 0
    outofbound = ((min - x1)[bad] > 0) | ((max - x1)[bad] <= 0)
    Tbad = numpy.tile(T[..., None], (1, 3))[bad]
    
    t1[bad] = numpy.where(outofbound, Tbad + 1, -1)
    t2[bad] = numpy.where(outofbound, -1, Tbad + 1)

    tmin = numpy.where(Dx >= 0, t1, t2)
    tmax = numpy.where(Dx >= 0, t2, t1)
    tmin = tmin.max(axis=-1)
    tmax = tmax.min(axis=-1)

    tmin[tmin < 0] = 0
    tmin[tmin > T] = T[tmin > T] + 1
    tmax[tmax < 0] = -1
    tmax[tmax > T] = T[tmax > T]

    #print t1, t2, tmin, tmax, numpy.sign(Dx) * -1
    # t range from tmin to tmax, inclusive

    Npixels = tmax - tmin + 1
    Npixels[Npixels < 0] = 0

    if not return_full:
        return Npixels

    cumsum = Npixels.cumsum()
    offset = numpy.concatenate(([0], cumsum[:-1]))

    N = cumsum[-1]
    o = numpy.repeat(numpy.arange(len(Dx)), Npixels)
    t = numpy.arange(N) - (offset - tmin)[o]
    #print t[0], tmin, tmax, t1, t2, T, Dx
    #print (min - x1) * T, Dx, (min - x1) * T * 1.0 / Dx

    x = t[..., None] * Dx[o]
    x //= T[o][..., None]
    x += x1[o]
    # remove the edge points
    return x, o, t

def test():
    v = numpy.array(([0, 0], [0, 2], [2, 0], [2, 2]))

    x1 = [  9968.60423495 ,17406.97619585 ,40308.06971767]
    x2 = [  7771.39808555 ,15926.61521123 ,41034.71562185]
    min = [    0 ,8192,40960]
    max = [m + 8129 for m in min]

    print clipline([x1], [x2], min, max, return_full=True)[0]

    tests = {
        ((0, 0), (0, 0)): 1,
        ((0, 0), (2, 2)): 2,
        ((1, 1), (2, 2)): 1,
        ((-10, -10), (2, 2)): 2,
        ((0, -10), (2, 2)): 2,
        ((0, -1), (2, 2)): 2,
        ((0, -1), (2, 3)): 2,
        ((0, 2), (2, 2)): 0,
        ((0, 0), (2, 0)): 2,
        }
    for x1, x2 in tests:
        n = tests[(x1, x2)]
        print x1, x2, n
        assert n == clipline([x1], [x2], (0, 0), (2, 2), return_full=False)
        assert n == clipline([x2], [x1], (0, 0), (2, 2), return_full=False)
        
if __name__ == '__main__':
    test()
