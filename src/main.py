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

  if A.mode == 'sightlines':
    cmd.sightlines.main(A)
  if A.mode == 'gaussian':
    cmd.gaussian.main(A)
  if A.mode == 'lognormal':
    cmd.lognormal.main(A)
  if A.mode == 'matchmeanflux':
    cmd.matchmeanflux.main(A)
  if A.mode == 'makespectra':
    cmd.makespectra.main(A)
  if A.mode == 'export':
    cmd.export.main(A)
  if A.mode == 'qsocorr':
    cmd.qsocorr.main(A)

main()

