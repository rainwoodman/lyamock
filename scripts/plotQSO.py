import sys
sys.path.append('../')

from fit.common import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.gridspec import GridSpec
from glob import glob
from sys import argv

def plotdir(dir):
    config = Config(dir + '/paramfile', basedir=dir)
    sightlines = Sightlines(config)
    print dir, len(sightlines), QSODensityModel(config).Nqso

for dir in argv[1:]:
    try:
        plotdir(dir)
    except Exception as e: 
        print e
        continue
