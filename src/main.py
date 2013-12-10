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
#      matchmeanF (cheap, don't run with too many cores)
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

  fullname = 'commands.' + A.command
  module = __import__(fullname, fromlist=['main'])
  module.main(A)

main()

