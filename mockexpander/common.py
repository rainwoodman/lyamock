import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from lib.common import *
from lib.common import Config as ConfigBase
import sharedmem
from lib.chunkmap import chunkmap

class Config(ConfigBase):
    def __init__(self, paramfile, basedir=None):
        """ if given basedir will be prefixed to datadir """
        ConfigBase.__init__(self, paramfile)

        export = self.export
        self.export("MockExpander", [ "QuasarVACFile"])
        self.export("MockExpander", "ConsiderNeighbours", type=float,
                default=32)
        self.export("MockExpander", "RedshiftImportance", type=float,
                default=1e-9)
