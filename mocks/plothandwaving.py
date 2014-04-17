import sys
sys.path.append('../')

from fit.common import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.gridspec import GridSpec
from glob import glob
from sys import argv
from gaepsi.tools.subplots import subplots

def mean(dirs):
    f = [
            CorrFuncCollection(numpy.load(dir +
                '/datadir/bootstrap.npz')['red'])
            for dir in dirs
    ]
    ref = f[0].copy()
    err = f[0].copy()
    xi = numpy.mean([a.compress() for a in f], axis=0)
    std = numpy.std([a.compress() for a in f], axis=0)
    ref.uncompress(xi)
    err.uncompress(std)
    return ref, err

fig = Figure((6, 6))
gs = GridSpec(3, 2, hspace=0.05, wspace=0.05, left=0.16, right=0.85)
ax = subplots(fig, gridspec=gs, sharex='all', sharey='False')

dirs = sorted(list(argv[1:])) #glob('test*/')))
print dirs
config = Config(dirs[0] + '/paramfile')
cos = config.cosmology
D = cos.D(1 / 3.25)
f = numpy.load(dirs[0] + '/datadir/bootstrap.npz')
e = EigenModes(f['eigenmodes'])
colors = ['o'] * 100

mean, error = mean(dirs)
"""
b_F -0.23070017646 0.00689506418833
b_Q 4.38743034349 0.241887020186
beta_F 0.536726725029 0.0353746208622
beta_Q -0.0614971733952 0.0489810139739
"""
parameters = (-0.230 * D, 4.39 * D, 0.536, -0.06)
# qso
for i in range(3):
    linear = e(parameters)
    ax[i, 0].errorbar(
            mean[i].r * 1e-3,
            linear[i].r ** 2 * 1e-6 * 
            mean[i].monopole, 
            linear[i].r ** 2 * 1e-6 * 
            error[i].monopole, label='mocks')

    ax[i, 0].plot(
            linear[i].r * 1e-3,
            linear[i].r ** 2 * 1e-6 * 
            linear[i].monopole,
            'k', lw=2, label=r'linear theory (handwaving fit)')
    ax[i, 1].errorbar(
            mean[i].r * 1e-3,
            linear[i].r ** 2 * 1e-6 * 
            mean[i].quadrupole, 
            linear[i].r ** 2 * 1e-6 * 
            error[i].quadrupole, label='mocks')

    ax[i, 1].plot(
            linear[i].r * 1e-3,
            linear[i].r ** 2 * 1e-6 * 
            linear[i].quadrupole,
            'k', lw=2, label=r'hand-waving theory')

    x1, x2 = ax[i, 0].get_ylim()
    x1 = max(numpy.abs([x1, x2]))
    ax[i, 0].set_ylim(-x1, x1)

    x1, x2 = ax[i, 1].get_ylim()
    x1 = max(numpy.abs([x1, x2]))
    ax[i, 1].set_ylim(-x1, x1)

    ax[i, 0].yaxis.tick_left()
    ax[i, 1].yaxis.tick_right()
    ax[i, 1].yaxis.set_label_position('right')
    ax[i, 0].set_ylabel(r'$r^2 \xi_0(r)$')
    ax[i, 1].set_ylabel(r'$r^2 \xi_2(r)$')
    ax[i, 0].axvspan(0, 20, hatch='//', color='gray')
    ax[i, 1].axvspan(0, 20, hatch='//', color='gray')
ax[i, 0].set_xlim(0, 160)
ax[i, 0].set_xticks([20, 50, 100, 150])
ax[i, 0].set_xlabel(r'$r\,h^{-1}\mathrm{Mpc}$')
ax[i, 1].set_xlabel(r'$r\,h^{-1}\mathrm{Mpc}$')


ax[2, 1].legend(fontsize='small', frameon=False)
#ax.prune_ticks()

#ax1.set_ylim(0, 1.5)
#ax1.set_xscale('log')
#ax1.set_yscale('log')
canvas = FigureCanvasAgg(fig)
#fig.tight_layout()
fig.savefig('handwaving.png')
fig.savefig('handwaving.svg')
