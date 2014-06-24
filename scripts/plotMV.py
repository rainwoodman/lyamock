import sys
sys.path.append('../')

from fit.common import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.gridspec import GridSpec
from glob import glob
from sys import argv
from gaepsi.tools.subplots import subplots

def plotdir(dir, ax1, ax2):
    f = numpy.load(dir + '/datadir/MeasuredMeanFractionOutput.npz')
    config = Config(dir + '/paramfile', basedir=dir)
    z = 1 / f['a'] - 1
    a = f['a']
    xmeanF = f['xmeanF']
    xvarF = f['xvarF']
    mean = MeanFractionModel(config)
    var = VarFractionModel(config)

    ax1.plot(z, xmeanF, '+')
    ax1.plot(z, mean(a), '-', label=r'$\left<F\right>$')
    ax2.plot(z, xvarF, 'x')
    ax2.plot(z, var(a), '-', label=r'$\sigma_F$')

fig = Figure((4, 4))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plotdir('000', ax1, ax2)
ax1.set_ylabel(r'$\left<F\right>(z)$')
ax2.set_ylabel(r'$\sigma_F(z)$')
ax1.set_xlabel(r'$z$')
ax2.set_xlabel(r'$z$')

ax1.legend(frameon=False, fontsize='small')
ax2.legend(frameon=False, fontsize='small')
canvas = FigureCanvasAgg(fig)
fig.tight_layout()
fig.savefig('MV.svg')
fig.savefig('MV.png')
