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
    print len(f)
    ref = f[0].copy()
    err = f[0].copy()
    fac = 1. / len(dirs) ** 0.5
    for i in range(3):
        xi = numpy.mean([a[i].poles for a in f], axis=0)
        std = numpy.std([a[i].poles for a in f], axis=0) * fac
        ref[i].frompoles(xi)
        err[i].frompoles(std)

    print err[0].monopole
    print err[0].poles[:, 0]
    return ref, err

fig = Figure((6, 6))
gs = GridSpec(6, 2, hspace=0.05, wspace=0.05, left=0.16, right=0.85)
ax = subplots(fig, gridspec=gs, sharex='all', sharey='False')
ax = ax.reshape(3, 2, 2)
ax1 = ax[:, 0, :]
ax2 = ax[:, 1, :]

dirs = sorted(list(argv[1:])) #glob('test*/')))
print dirs
config = Config(dirs[0] + '/paramfile')
cos = config.cosmology
D = cos.D(1 / 3.25)
f = numpy.load('eigenmodes.npz')
e = EigenModes(f['eigenmodes'])
colors = ['o'] * 100

mean, error = mean(dirs)
error2 = CorrFuncCollection(numpy.load('fit.npz')['error'])
fits = numpy.load('fit.npz')['fittedparameters']
parameters = (
        -numpy.abs(fits[:, 0]).mean(),
        numpy.abs(fits[:, 1]).mean(),
        fits[:, 2].mean(),
        fits[:, 3].mean())
print parameters
# qso
for i in range(3):
  for j, order in enumerate([0, 2]):
    linear = e(parameters)
    print i, j, order
    ax1[i, j].errorbar(
            mean[i].r * 1e-3,
            linear[i].r ** 2 * 1e-6 * 
            mean[i].poles[:, order], 
            linear[i].r ** 2 * 1e-6 * 
            error[i].poles[:, order], label='mocks')

    ax1[i, j].errorbar(
            mean[i].r * 1e-3,
            linear[i].r ** 2 * 1e-6 * 
            mean[i].poles[:, order], 
            linear[i].r ** 2 * 1e-6 * 
            error2[i].poles[:, order] / len(dirs) ** 0.5, 
            label='mocks from cov')

    ax1[i, j].plot(
            linear[i].r * 1e-3,
            linear[i].r ** 2 * 1e-6 * 
            linear[i].poles[:, order],
            'k', lw=2, label=r'linear theory (handwaving fit)')

    ax2[i, j].plot(
            linear[i].r * 1e-3,
            linear[i].r ** 2 * 1e-6 * 
            (mean[i].poles[:, order] - linear[i].poles[:, order]))

    ax2[i, j].errorbar(
            linear[i].r * 1e-3,
            numpy.zeros_like(linear[i].r),
            linear[i].r ** 2 * 1e-6 * 
            error2[i].poles[:, order] / len(dirs) ** 0.5)

    x1, x2 = ax1[i, j].get_ylim()
    x1 = max(numpy.abs([x1, x2]))
    ax1[i, j].set_ylim(-x1, x1)

    if j == 0:
        ax1[i, j].yaxis.tick_left()
        ax1[i, j].set_ylabel(r'$r^2 \xi_0(r)$')
    else:
        ax1[i, j].yaxis.tick_right()
        ax1[i, j].yaxis.set_label_position('right')
        ax1[i, j].set_ylabel(r'$r^2 \xi_2(r)$')
    ax1[i, j].axvspan(0, 20, hatch='//', color='gray')

ax2[-1, 0].set_xlim(0, 160)
ax2[-1, 0].set_xticks([20, 50, 100, 150])
ax2[-1, 0].set_xlabel(r'$r\,h^{-1}\mathrm{Mpc}$')
ax2[-1, 1].set_xlabel(r'$r\,h^{-1}\mathrm{Mpc}$')


ax2[-1, 1].legend(fontsize='small', frameon=False)
canvas = FigureCanvasAgg(fig)
fig.savefig('handwaving.png')
fig.savefig('handwaving.svg')
