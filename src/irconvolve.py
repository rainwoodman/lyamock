import numpy
from scipy.sparse import csr

def irconvolve(xc, x, y, h, 
        kernel=lambda r, h: numpy.exp(- 0.5 * (r / h) ** 2)):
    """ default kernel is gaussian
        exp - 1/2 * r / h
        xc has to be uniform!
    """
    xc, y, x, h = numpy.atleast_1d(xc, y, x, h)
    dxc = (xc[-1] - xc[0]) / (len(xc) - 1)
    support = 6

    #first remove those are too far off
    good = ((x + support * h > xc[0]) \
          & (x - support * h < xc[-1]))
    x = x[good]
    y = y[good]
    h = h[good]

    out = numpy.zeros(shape=xc.shape, dtype=y.dtype)

    # the real buffer is bigger than out to ease the normalization
    # still on the edge we are imperfect
    padding = int((2 * support + 1)* h.max() / dxc) + 1
    padding = max(padding, 2)
    buffer = numpy.zeros(shape=len(xc) + 2 * padding)
    paddedxc = numpy.empty(buffer.shape, dtype=xc.dtype)
    paddedxc[padding:-padding] = xc
    # here comes the requirement xc has to be uniform.
    paddedxc[:padding] = xc[0] - numpy.arange(padding, 0, -1) * dxc
    paddedxc[-padding:] = xc[-1] + numpy.arange(1, padding +1) * dxc
    out = buffer[padding:-padding]
    assert len(out) == len(xc)
    assert (paddedxc[1:] > paddedxc[:-1]).all()

    # slow. for uniform xc/paddedxc, we can do this faster than search
    start = paddedxc.searchsorted(x - support * h, side='left')
    end = paddedxc.searchsorted(x + support * h, side='left')

    # tricky part, build the csr matrix for the conv operator,
    # only for the non-zero elements (block diagonal)
    N = end - start + 1
    indptr = numpy.concatenate(([0], N.cumsum()))
    indices = numpy.repeat(start - indptr[:-1], N) + numpy.arange(N.sum())
    r = numpy.repeat(x, N) - paddedxc[indices]
    data = kernel(r, numpy.repeat(h, N))
    data[numpy.repeat(N==1, N)] = 1
    data[numpy.repeat(h==0, N)] = 1
    matrix = csr.csr_matrix((data, indices, indptr), 
            shape=(len(x), len(paddedxc)))
    norm = numpy.repeat(matrix.sum(axis=1).flat, N)
    data /= norm
    buffer[:] = matrix.transpose() * y
    return out

def test():
#    x = numpy.random.uniform(size=100)
    x = numpy.linspace(0, 1, 40)
    y = numpy.sin(x)
    xc = numpy.linspace(0, 1, 400)
    h = 0. * numpy.ones_like(x)
    yc = irconvolve(xc, x, y, h)
    plot(x, y, '. ')
    plot(xc, yc)
    print y, yc
    print y.sum(), yc.sum()

