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

def integer_ineq(x, y, gl, eq):
    """ 
        find extreme value of 
        gl == 1, eq==1: 
           t * y >= x
        gl == -1 eq==1:
           t * y <= x
        gl == 1, eq==0: 
           t * y > x
        gl == -1 eq==0:
           t * y < x
    """
    r = x / y
    # flip the sign
    newgl = numpy.sign(y) * gl
    # transform to t ?? x / y
    rt = numpy.where(newgl, 
            numpy.ceil(r),
            numpy.floor(r))
    rt += numpy.where((eq == 0) & (r == rt),
            newgl, 0)
    return rt

def drawline(x1, x2, sep, min=None, max=None, return_full=False):
    """ draw a line from x1 to (at least) x2, with pixel separation
        sep. only preserve points in the box given by >=min and < max.
        x1, x2 can be a series of lines. 
        (this is not optimized yet, still using an explicit loop)
    """
    x1, x2 = numpy.atleast_2d(x1, x2)
    dir = x2 - x1
    L = ((x2 - x1) ** 2).sum(axis=-1) ** 0.5
    bad = L == 0
    L[bad] = 1.0
    dir = dir * ((1.0 * sep) / L[:, None])
    L[bad] = 0.0
    rt = []

    tmin, tmax = fliangbaskey(x1, x2, min, max)
    Nmin = numpy.int32(tmin * L / sep - 1) 
    Nmin[Nmin < 0] = 0
    Nmax = numpy.int32(tmax * L / sep + 1)
    bad = Nmax > L / sep + 1
    Nmax[bad] = L[bad] / sep + 1
    Npixels = Nmax - Nmin + 1
    Npixels[L == 0] = 1
    Npixels[Npixels < 0] = 0

    cumsum = Npixels.cumsum()
    offset = numpy.concatenate(([0], cumsum[:-1]))

    N = cumsum[-1]
    o = numpy.repeat(numpy.arange(len(x1)), Npixels)
    t = numpy.arange(N) - (offset - Nmin)[o]
    #print t[0], tmin, tmax, t1, t2, T, Dx
    #print (min - x1) * T, Dx, (min - x1) * T * 1.0 / Dx

    x = (t[..., None] * dir[o])
    x += x1[o]
    if len(x) > 0:
        goodmask = numpy.all(x >= min, axis=-1)
        goodmask &= numpy.all(x < max, axis=-1)
        x = x[goodmask]
        o = o[goodmask]
    Npixels[:] = numpy.bincount(o, minlength=len(Npixels))
    if not return_full:
        return Npixels
    else:
        #print 't', t, 'dir', dir[o]
        return x, o, None

def fliangbaskey(x1, x2, min, max):
    """
        do the clipping with liang baskey, returns
        tmin, tmax

        tmin, tmax are in range(0, 1) (0 at x1, 1 at x2)
    """
    x1 = numpy.asarray(x1)
    x2 = numpy.asarray(x2)
    Dx = x2 - x1
    bad = Dx == 0
    #protect against 0
    vec = Dx
    vec[bad] = 1
    # we want to round towards 0
    min = numpy.array(min, dtype='f8')
    max = numpy.array(max, dtype='f8')
    t1 = (min - x1) / vec
    t2 = (max - x1) / vec
    vec[bad] = 0
    outofbound = ((min - x1)[bad] > 0) | ((max - x1)[bad] <= 0)
    
    t1[bad] = numpy.where(outofbound, 100, -100)
    t2[bad] = numpy.where(outofbound, -100, 100)

    tmin = numpy.where(vec >= 0, t1, t2)
    tmax = numpy.where(vec >= 0, t2, t1)
    tmin = tmin.max(axis=-1)
    tmax = tmax.min(axis=-1)

    tmin[tmin < 0] = 0
    tmin[tmin > 1] = 1
    tmax[tmax < 0] = 0
    tmax[tmax > 1] = 1

    #print t1, t2, tmin, tmax, numpy.sign(Dx) * -1
    # t range from tmin to tmax, inclusive
    return tmin, tmax

def liangbaskey(x1, x2, min, max):
    """
        do the clipping with liang baskey, returns
        tmin, tmax, T

        the line goes from 0 to T; T, tmin, tmax are integers,
        rounded to the nearest integer.
        x1, x2, min, max must be all integers.
    """
    x1 = numpy.asarray(x1)
    assert x1.dtype.kind == 'i'
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
    return tmin, tmax, T

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

    tmin, tmax, T = liangbaskey(x1, x2, min, max)
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

def testclipline():
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
        
def testdrawline():
    print drawline([0, 0], [10, 10], 1, [2, 2], [5, 5], True)
    print fliangbaskey([[0, 0]], [[10, 10]], [2, 2], [5, 5])
    tests = {
        ((0, 0), (0, 0)): 1,
        ((0, 0), (2, 2)): 3,
        ((2, 2), (0, 0)): 2,
        ((1, 1), (2, 2)): 2,
        ((2, 2), (1, 1)): 2,
        ((-10, -10), (2, 2)): 2,
        ((0, -10), (2, 2)): 2,
        ((0, -1), (2, 2)): 2,
        ((0, -1), (2, 3)): 2,
        ((0, 2), (2, 2)): 0,
        ((2, 2), (0, 2)): 0,
        ((0, 0), (2, 0)): 2,
        }
    for x1, x2 in tests:
        n = tests[(x1, x2)]
        print x1, x2, n
        print drawline([x1], [x2], 1.0, (0, 0), (2, 2), return_full=True)
        print  drawline([x1], [x2], 1.0, (0, 0), (2, 2),
                return_full=False)[0]
        assert n == drawline([x1], [x2], 1.0, (0, 0), (2, 2),
                return_full=False)[0]
if __name__ == '__main__':
    testdrawline()
