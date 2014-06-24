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
    f = numpy.load(dir + '/datadir/MatchMeanFractionOutput.npz')
    config = Config(dir + '/paramfile', basedir=dir)
    z = 1 / f['a'] - 1
    a = f['a']
    Af = f['Af']
    Bf = f['Bf']
    model = FGPAmodel(config)
    A = model.Afunc
    B = model.Bfunc
    ax1.plot(z, 1000 * Af, '+')
    ax1.plot(z, 1000 * A(a), '-', label='exp fit')
    ax2.plot(z, Bf, 'x')
    ax2.plot(z, B(a), '-', label='2nd-poly fit')

fig = Figure((4, 4))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plotdir('000', ax1, ax2)
ax1.set_ylabel(r'$1000 A(z)$')
ax2.set_ylabel(r'$B(z) \sim 2.0 - 0.7(\gamma - 1)$')
ax1.set_xlabel(r'$z$')
ax2.set_xlabel(r'$z$')

ax1.legend(frameon=False, fontsize='small')
ax2.legend(frameon=False, fontsize='small')
canvas = FigureCanvasAgg(fig)
fig.tight_layout()
fig.savefig('AB.svg')
fig.savefig('AB.png')
