import numpy
import sharedmem
from common import Config
from common import Sightlines
from common import FGPAmodel
from lib.chunkmap import chunkmap
from scipy.spatial import cKDTree
import chealpy
import fitsio
from sys import argv

def decra2vec(dec, ra, redshift):
    z = numpy.sin(dec)
    r = numpy.cos(dec)
    x = r * numpy.cos(ra)
    y = r * numpy.sin(ra)
    return numpy.array([x, y, z, redshift + 1]).T.copy()

def readquasars(A):
    q = Sightlines(A)
    return numpy.rec.fromarrays([q.DEC, q.RA, q.Z_RED], names=['DEC', 'RA', 'Z'])

def readparis(A):
    with fitsio.FITS(A.QuasarVACFile) as cat:
        hdu = cat[1]
        mask = hdu['THING_ID'][:] > 0
        dec = hdu['DEC'][:][mask] * (numpy.pi / 180)
        ra = hdu['RA'][:][mask] * (numpy.pi / 180)
        ra[ra > numpy.pi] -= numpy.pi * 2
        plate = hdu['plate'][:][mask]
        mjd = hdu['mjd'][:][mask]
        fiber = hdu['fiberid'][:][mask]
        z_vi = hdu['Z_VI'][:][mask]
    return numpy.rec.fromarrays([dec, ra, z_vi, plate, mjd, fiber],
            names=['DEC', 'RA', 'Z', 'PLATE', 'MJD', 'FIBER'])

def stat(name, dataset, fixed=True):
    from numpy import histogram
    print 'stat of data set', name
    rah, rab = histogram(dataset.RA, range=None if not fixed else (-numpy.pi, numpy.pi), bins=20)
    dech, decb = histogram(dataset.DEC, range=None if not fixed else (-numpy.pi * 0.5, numpy.pi * 0.5), bins=20)
    zh, zb = histogram(dataset.Z, range=None if not fixed else (0, 6.0), bins=20)
    if not fixed:
        print 'ra bins', rab
    print 'ra', 'std', numpy.std(dataset.RA), 'histogram', rah
    if not fixed:
        print 'dec bins', decb
    print 'dec', 'std', numpy.std(dataset.DEC), 'histogram', dech
    if not fixed:
        print 'z bins', zb
    print 'z', 'std',numpy.std(dataset.Z), 'histogram', zh

def checkuniqueness(matchupresult):
    stride2 = numpy.max(matchupresult.FIBER)
    stride1 = numpy.max(matchupresult.MJD) * stride2
    UID = matchupresult.PLATE * stride1 + \
            matchupresult.MJD * stride2 + matchupresult.FIBER

    u, ri = numpy.unique(UID, return_inverse=True)
    degen = numpy.bincount(ri)
    worst = degen.argmax()
    print 'max degeneracy is', degen.max()
    #print 'these guys all matched to one fiber'
    #print matchupresult[ri == worst]
    degenh = numpy.bincount(degen)
    print 'conflict / total', len(UID) - len(numpy.unique(UID)), '/', len(UID)
    print 'degeneracy', degenh
def randomanycolumn(ind, rng=numpy.random):
    # 2d input ind, pick random element along last d
    # make sure it is continous
    if len(ind.shape) == 1: ind = ind.reshape(-1, 1)
    ind = ind.copy()
    nd = ind.shape[-1]
    ne = ind.shape[0]
    sel = numpy.arange(ne) * nd + numpy.arange(ne) % nd
    #rng.randint(nd)
    return ind.flat[sel] 

def resolveconflict(ind):
    """heuristicly pick fibers that resolves the conflicts,,

       assign quasar to the nearest fiber

       find conflicts
       for conflicts,
          move to a 'semi'-random neighbour in the nearest list
       go back find conflicts

       try it a few times and report the best solution with
       the least number of conflicts

    """
    print 'resolving conflicts'
    if len(ind.shape) == 1: ind = ind.reshape(-1, 1)

    ne, nd = ind.shape
    # currently used item in ind
    pick = numpy.zeros(len(ind), dtype='i4')
    pick[:] = numpy.arange(ne) % nd
    bestpick = None
    bestunique = 0
    bestdegen = 999999
    i = 0
    while i < 100:
        # currently used item
        current = ind.flat[pick + numpy.arange(ne) * nd]
        junk, id, ri = numpy.unique(current, return_index=True, return_inverse=True)
        print 'conflict / all', ne - len(junk), '/', ne, 'iteration = ', i
# new algorithm
        counts = numpy.bincount(ri) # for each id

        if len(junk) > bestunique or bestdegen > counts.max():
            bestunique = len(junk)
            bestpick = pick.copy()
            bestdegen = counts.max()
            i = 0
        nonunique = id[counts > 1]
        if False: # i == 0:
#            uniquechanged = numpy.random.choice(id[counts == 1], size=10,
#                    replace=False)
            uniquechanged = id[counts == 1][len(nonunique)%100::100]
            print len(uniquechanged), len(nonunique)
            changed = numpy.concatenate([nonunique, uniquechanged])
        else:
            changed = nonunique
        changed.sort()
        pick[changed] += numpy.arange(len(changed))
        pick[changed] %= nd
        i = i + 1
# old algorithm
#        counts = numpy.bincount(ri)[ri]
#        nonuniquemask = counts > 1
#        pick[nonuniquemask] += numpy.arange(nonuniquemask.sum())
#        pick[nonuniquemask] %= nd

    current = ind.flat[bestpick + numpy.arange(ne) * nd]
    return current

def main(A):
    quasars = readquasars(A)
    paris = readparis(A)

    #selection = numpy.arange(0, len(quasars), 1000)
    #quasars = quasars[selection]

    mockpos = decra2vec(quasars.DEC, quasars.RA, quasars.Z * A.RedshiftImportance)
    parispos = decra2vec(paris.DEC, paris.RA, paris.Z * A.RedshiftImportance)
    
    stat('quasars', quasars)
    stat('paris', paris)

    lookup = cKDTree(parispos)
    d, ind = lookup.query(mockpos, k=A.ConsiderNeighbours)

    ind = resolveconflict(ind)
    
    matchup = paris[ind]

    diff = matchup.copy()
    diff.DEC[:] = matchup.DEC - quasars.DEC
    diff.RA[:] = matchup.RA - quasars.RA
    diff.RA[diff.RA > numpy.pi] -= numpy.pi * 2
    diff.RA[diff.RA < -numpy.pi] += numpy.pi * 2
    diff.Z[:] = matchup.Z - quasars.Z
    stat('diff', diff, fixed=False)
    matchupresult = numpy.rec.fromarrays(
            [
            numpy.arange(len(quasars)),
            quasars.DEC, quasars.RA, quasars.Z,
            matchup.DEC, matchup.RA, matchup.Z,
            matchup.PLATE, matchup.FIBER, matchup.MJD,
            ],
            names=[
                'Q_INDEX', 
                'Q_DEC', 'Q_RA', 'Q_Z',
                'R_DEC', 'R_RA', 'R_Z',
                'PLATE', 'FIBER', 'MJD'
            ]
        )
    matchupresult.sort(order=['PLATE', 'FIBER', 'MJD'])
    checkuniqueness(matchupresult)
    numpy.save('matchup-result.npy', matchupresult)

if __name__ == '__main__':
    main(Config(argv[1]))
