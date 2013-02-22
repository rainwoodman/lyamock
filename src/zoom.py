import math
import numpy
from scipy.ndimage import _ni_support
from scipy.ndimage import _nd_image
from scipy.ndimage.interpolation import spline_filter
from scipy.ndimage.interpolation import _extend_mode_to_code
def zoom(input, zoom, output=None, order=3, mode='constant', cval=0.0,
         prefilter=True):
    """
    Zoom an array.

    The array is zoomed using spline interpolation of the requested order.

    Parameters
    ----------
    input : ndarray
        The input array.
    zoom : float or sequence, optional
        The zoom factor along the axes. If a float, `zoom` is the same for each
        axis. If a sequence, `zoom` should contain one value for each axis.
    output : ndarray or dtype, optional
        The array in which to place the output, or the dtype of the returned
        array.
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        Default is 'constant'.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0
    prefilter : bool, optional
        The parameter prefilter determines if the input is pre-filtered with
        `spline_filter` before interpolation (necessary for spline
        interpolation of order > 1).  If False, it is assumed that the input is
        already filtered. Default is True.

    Returns
    -------
    return_value : ndarray or None
        The zoomed input. If `output` is given as a parameter, None is
        returned.

    """
    if order < 0 or order > 5:
        raise RuntimeError('spline order not supported')
    input = numpy.asarray(input)
    if numpy.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    if input.ndim < 1:
        raise RuntimeError('input and output rank must be > 0')
    mode = _extend_mode_to_code(mode)
    if prefilter and order > 1:
        filtered = spline_filter(input, order, output = numpy.float64)
    else:
        filtered = input
    zoom = _ni_support._normalize_sequence(zoom, input.ndim)
    output_shape = tuple([int(ii * jj) for ii, jj in zip(input.shape, zoom)])

    zoom_div = numpy.array(output_shape, float)
    zoom = (numpy.array(input.shape)) / zoom_div

    # Zooming to non-finite values in unpredictable, so just choose
    # zoom factor 1 instead
    zoom[~numpy.isfinite(zoom)] = 1

    output, return_value = _ni_support._get_output(output, input,
                                                   shape=output_shape)
    zoom = numpy.asarray(zoom, dtype = numpy.float64)
    zoom = numpy.ascontiguousarray(zoom)
    _nd_image.zoom_shift(filtered, zoom, None, output, order, mode, cval)
    return return_value
