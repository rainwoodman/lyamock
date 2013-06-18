import numpy

def clipline(x1, x2, dims, return_full=True, return_lineid=False):
    """return integer positions
       of line from x1 to x2 on a grid of size dims.

       x1 and x2 must be integers (no subpixel stuff)
       pixels are weighted on their topleft corner.

       x1 is a row vector and x2 is a row vector.
       if return_full == False,
          returns total number of intersecting pixels per line
       otherwise, continue reading:

       returns a column vector.
          return X, Y, Z, ....

       if return_lineid == True, also
       returns the index of the line and the index of the
       pixel in the line.
          return (X, Y, Z, ..), (Line, Pixel)

          Wikipedia Liang-Barsky.

          u is u / absdxpri
    """

    # parametrize along the primary axis
    # x = x1 + (xp - xp1) / dxp * dx
    # 
    x1, x2 = numpy.atleast_2d(numpy.int32(x1), numpy.int32(x2))
    dims = numpy.int32(dims)
    dx = x2 - x1
    pri = numpy.abs(dx).argmax(axis=-1)

    dxpri = numpy.atleast_1d(numpy.choose(pri, numpy.rollaxis(dx, -1)))
    absdxpri = numpy.abs(dxpri)
    slope = dx * numpy.sign(dxpri)[:, None]
    Np = len(slope)
    Nd = slope.shape[-1]

    p = numpy.empty(list(slope.shape[:-1]) + [Nd,  2])
    q = numpy.empty_like(p)
    size = numpy.zeros(Np, dtype=numpy.intp)

    q[..., 0] = x1
    q[..., 1] = dims - x1
    q *= absdxpri[..., None, None]

    p[..., 0] = - dx
    p[..., 1] = dx

    qp = q / p

    # outside->in
    u1 = numpy.where(p < 0, qp, -1)
    u1 = u1.max(axis=(-1, -2))
    u1[u1 < 0] = 0
    u1 = numpy.ceil(u1)
    # inside->out
    u2 = numpy.where(p > 0, qp, absdxpri[..., None, None])
    u2 = u2.min(axis=(-1, -2))
    u2[u2 > absdxpri] = absdxpri[u2 > absdxpri]
    u2 = numpy.floor(u2)
    excl = ((p == 0) & (q < 0)).any(axis=(-1, -2)) | (dxpri == 0)
    u1[excl] = u2[excl] + 1
    size[u1 <= u2] = (u2 + 1 - u1)[u1 <= u2]

    if not return_full:
        return size

    #delete unused variables, leaving memory for result
    offset = numpy.concatenate(([0], size.cumsum()[:-1]))
#    p, q, qp, u2 = None, None, None, None

    N = size.sum()

    result = numpy.empty((N, Nd), numpy.int32)
    if return_lineid:
      lineid, pixelid = numpy.zeros((2, N), numpy.intp)
      if len(result) == 0:
        return result.T, lineid, pixelid
    else:
      if len(result) == 0:
        return result.T

    for i in range(0, Np, 128):
        S = slice(i, i+128)
        R = slice(offset[i], offset[i]+size[S].sum())
        u0 = numpy.repeat(offset[S] - u1[S], size[S])
        u = offset[i] + numpy.arange(size[S].sum()) - u0
        sl = numpy.repeat(numpy.float64(slope[S]) / dxpri[S][..., None],
                size[S], axis=0)
        xx1 = numpy.repeat(x1[S], size[S], axis=0)
        assert not numpy.isnan(sl).any()
        result[R] = numpy.int32(u[:, None] * sl + xx1)
        if return_lineid:
            lineid[R] = numpy.repeat(numpy.arange(*S.indices(Np)), 
                size[S])
            pixelid[R] = u

    result = result.T
    mask = ~(result >= dims[:, None]).any(axis=0)
    mask &= ~(result < 0).any(axis=0)
    # remove the edge points
    if return_lineid:
        return result[:, mask], lineid[mask], pixelid[mask]
    else:
        return result[:, mask]
