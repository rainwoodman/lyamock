import numpy

def snakefill(shape):
    """ return the snake fill index of an array.

        shape must be even.

        along the first Ndim - 1 axes,
        k_d goes from 0 ..., N / 2, - N / 2 + 1, ... -1.
        along the last axes,
        k_d goes from 0 ... N
        where N == shape[d]
        the returned snake fill index is not continues
        but guarenteed in increasing order.

        to get the correct snake index 
        for the flatened array, use
        snakefill(array).flat.argsort()
    """
    A = numpy.zeros(shape, 'i8')
    istart = 0
    for n in range(shape[0] // 2 + 1):
        istart = makeseq(n, A, istart)
    return A

def comb(*args):
    """ return combinations, selecting
        one item from each of its args

        eg,
        comb([0, 1], ['a', 'b'])

        returns four items 
        0, a
        1, a
        0, b
        1, b
    """
    if len(args) == 1:
        for a in args[0]:
            yield [a]
        return
    for a in args[0]:
        for t in comb(*(args[1:])):
            yield [a] + t


def makeseq(n, A, istart):
    """ fill the nth shell of the snake fill
        the last dimension is non-periodic
        
    """
    shape = A.shape
    Ndim = len(A.shape)
    NEGAX = lambda x: 0
    POSAX = lambda x: 0
    NEGPT = lambda x: 0
    POSPT = lambda x: 0
    c = NEGPT, POSPT, NEGAX, POSAX
    if n > shape[0] / 2: return
    def length(set):
        """ returns the number of points in the
            indexing 'set'. duplicated are double
            counted.
        """
        rt = 1
        for d, x in enumerate(set):
            if not isinstance(x, slice): continue
            start, end, step = x.indices(shape[0])
            rt *= (end - start)
        return rt
    def translate(set):
        """translate the index set 
            from NEGAX POSAX to real indexing
            elements
        """
        for d, x in enumerate(set):
            if x is POSAX:
              if n > 0:
                set[d] = slice(0, n)
              else:
                set[d] = slice(0, 0)
            elif x is NEGAX:
              assert d != Ndim - 1
              if n > 0:
                set[d] = slice(shape[d] - n, shape[d])
            elif x is NEGPT:
                assert d != Ndim - 1
                set[d] = shape[d] - n
            else:
                set[d] = n
    if n == 0:
       A.flat[0] = istart
       return istart + 1

    j = istart
    for points in comb(*([c] * Ndim)):
        if numpy.all([x is NEGAX or x is POSAX for x in points]):
            # skip inside
            continue
        if numpy.any([(x is NEGAX or x is NEGPT) and d == Ndim - 1 \
                for d, x in enumerate(points)]):
            # skip last ax if we are corrossing negative part
            continue
        translate(points)
        l = length(points)
        if l == 1:
            A[tuple(points)] = j
        else:
            A[tuple(points)].flat = range(j, j + l)
        j += l

    return j
