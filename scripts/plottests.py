import sys
sys.path.append('../')

from fit.common import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from glob import glob
from sys import argv

def plotdir(dir, ax1, ax2, linear, D, color):
    f = numpy.load(dir + '/datadir/bootstrap.npz')
    config = Config(dir + '/paramfile')
    QSOScale = config.QSOScale
    Nq = f['Qchunksize'].sum()
    bias = QSOBiasModel(config)(1.0 / 3.5)
    label = 'Nq = %d, Scale=%g, b=%g' % (Nq, QSOScale / 1000., bias)
    ax1.plot(
           linear[0].r ,
        f['red'][0].monopole / 
            #D ** 2 * 3.6 ** 2 * 
            (bias ** 2 * linear[0].monopole) ,
        color, label=label)
    ax2.plot(
           linear[0].r ,
#        linear[0].r ** 2 * 1e-6 * 
        f['red'][0].monopole,
        color, label=label)

fig = Figure()

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
dirs = sorted(list(argv[1:])) #glob('test*/')))
print dirs
config = Config(dirs[0] + '/paramfile')
cos = config.cosmology
D = cos.D(1 / 3.5)
f = numpy.load(dirs[0] + '/datadir/bootstrap.npz')
e = EigenModes(f['eigenmodes'])
linear = e((1.0, 1.0, 1.0, 1.0))
colors = [
        'rx', 'r+', 'r>', 'rv',
        'gx', 'gs', 'g+',
        'bx', 'bs', 'b+',]

for dir,color in zip(dirs, colors):
    plotdir(dir, ax1, ax2, linear, D, color)

ax1.legend(fontsize='small', frameon=False)
ax1.set_yscale('linear')
ax1.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xscale('log')
ax1.set_ylim(0, 1.5)
ax2.plot(
           linear[0].r ,
        3.6 ** 2 * D ** 2 *  
#        linear[0].r ** 2 * 1e-6 * 
         linear[0].monopole,
        'k', lw=2, label='MW ref xi')
ax2.plot(
           linear[0].r ,
         D ** 2 *  
#        linear[0].r ** 2 * 1e-6 * 
         linear[0].monopole,
         'k', lw=2, ls=':', label='linear theory')
ax2.set_ylim(0, 500)
canvas = FigureCanvasAgg(fig)
fig.savefig('tests.png')
