# Quasar Correlated Lyman alpha forest mocks
#
#  Mosaic (Nested Mesh) Mocks
#
#  Yu Feng 2013, Carnegie Mellon U
# 
#  input: a paramfile
#  run:
#      sightlines (if geometry in paramfile is not test)
#                 (less expensive, with as many cores as possible)
#      gaussian  (expensive, with as many cores as possible)
#      lognormal (cheap, with as few cores as possible)
#      matchmeanflux (cheap, don't run with too many cores)
#      makespectra (cheap, don't run with too many cores)
#
#  typical mock takes 2 hour on 16 core good machines, 
#              <30 min on 64 core
#  make sure fine mesh don't use more than memory per core.
#
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

