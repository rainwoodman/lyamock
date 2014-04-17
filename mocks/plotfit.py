import sys
sys.path.append('../')

from fit.common import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.gridspec import GridSpec
from glob import glob
from sys import argv
from gaepsi.tools.subplots import subplots

dirs = sorted(list(argv[1:])) #glob('test*/')))
print dirs
config = Config(dirs[0] + '/paramfile')
cos = config.cosmology
D = cos.D(1 / 3.25)
fit = numpy.load('fit.npz')

def plot(dir, ax):
    mock = int(dir)
    model = fit['models'][mock]
    error = fit['error']
    sample = fit['samples'][mock]

    for i in range(3):
      for j, pole in enumerate([0, 2]):
        ax[i, j].errorbar(
                sample[i].r * 1e-3,
                sample[i].r ** 2 * 1e-6 * 
                sample[i].poles[:, pole],
                sample[i].r ** 2 * 1e-6 * 
                error[i].poles[:, pole], label='mocks')

        ax[i, j].plot(
                model[i].r * 1e-3,
                model[i].r ** 2 * 1e-6 * 
                model[i].poles[:, pole],
                'k', lw=2, label=r'(best fit)')
        
        x1, x2 = ax[i, j].get_ylim()
        x1 = max(numpy.abs([x1, x2]))
        ax[i, 0].set_ylim(-x1, x1)


        ax[i, j].set_ylabel(r'$r^2 \xi_%d(r)$' % pole)
        if j == 0:
            ax[i, j].yaxis.tick_left()
        else:
            ax[i, j].yaxis.tick_right()
            ax[i, j].yaxis.set_label_position('right')
        ax[i, j].axvspan(0, 20, hatch='//', color='gray')

        ax[i, j].set_xlim(0, 160)
        ax[i, j].set_xticks([20, 50, 100, 150])
        ax[i, j].set_xlabel(r'$r\,h^{-1}\mathrm{Mpc}$')
    param = fit['fittedparameters'][mock]

    print param[0] / D, param[1] / D, param[2], param[3]

b_F = fit['fittedparameters'][:, 0] / D
b_Q = fit['fittedparameters'][:, 1] / D
beta_F = fit['fittedparameters'][:, 2] 
beta_Q = fit['fittedparameters'][:, 3] 

print 'b_F', b_F.mean(), b_F.std()
print 'b_Q', b_Q.mean(), b_Q.std()
print 'beta_F', beta_F.mean(), beta_F.std()
print 'beta_Q', beta_Q.mean(), beta_Q.std()

for dir in dirs:
    fig = Figure((6, 6))
    gs = GridSpec(3, 2, hspace=0.05, wspace=0.05, left=0.16, right=0.85)
    ax = subplots(fig, gridspec=gs, sharex='all', sharey='False')

    ax[2, 1].legend(fontsize='small', frameon=False)
    canvas = FigureCanvasAgg(fig)
    plot(dir, ax)
    fig.savefig(os.path.join(dir, 'bestfit.png'))
    fig.savefig(os.path.join(dir, 'bestfit.svg'))
