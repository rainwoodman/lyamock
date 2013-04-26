import numpy
from args import pixeldtype1, pixeldtype2
from args import writefill
import sharedmem

def main(A):
  """ check """
  def work(i, j, k):
    print A.basename(i, j, k, 'pass1')
    try:
        fill1 = numpy.fromfile(A.basename(i, j, k, 'pass1') + '.raw',
                pixeldtype1)
        assert (fill1['row'] >= 0).all()
        print 'good'
    except IOError as e:
        return

  with sharedmem.Pool() as pool:
    pool.starmap(work, list(A.yieldwork()))
