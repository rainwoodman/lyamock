import numpy
import commands as cmd
from args import parseargs

from splat import splat
from cosmology import interp1d
import sharedmem
#sharedmem.set_debug(True)

def main():
  numpy.seterr(all='ignore')
  A = parseargs()
  if A.serial:
      sharedmem.set_debug(True)

  if A.mode == 'firstpass':
    cmd.gaussian.main(A)
  if A.mode == 'secondpass':
    cmd.distort.main(A)
  if A.mode == 'thirdpass':
    cmd.meanfluxfactor.main(A)
  if A.mode == 'fourthpass':
    cmd.normalize.main(A)
  if A.mode == 'fifthpass':
    cmd.rebin.main(A)

main()

