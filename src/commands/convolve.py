import numpy
import sharedmem
from args import pixeldtype, bitmapdtype, sightlinedtype
from cosmology import interp1d
from irconvolve import irconvolve

def main(A):
    """convolve the tau(mass) field, 
    add in thermal broadening and redshift distortion """
    qsocatelog = A.P('QSOcatelog', memmap='r+', dtype=sightlinedtype)
    index = IndexBySightline(A)
    var = numpy.loadtxt(A.datadir + '/gaussian-variance.txt')
    print var
    # from Rupert
    # 0.12849 = sqrt( 2 KBT / Mproton) in km/s
    # He didn't have 0.5 in the kernel.
    # we do

    SQRT_KELVIN_TO_KMS = 0.12849 / 2. ** 0.5
    if len(index.ind) * 40 < sharedmem.total_memory():
        memmap = None
        print 'using memory'
    else:
        print 'using memmap'
        memmap = 'r'

    disp = A.P('losdisp', memmap=memmap)
    delta = A.P('delta', memmap=memmap)
    dc = A.P('dc', memmap=memmap)
    taureal = A.P('taureal', memmap='w+', shape=delta.shape)
    taured = A.P('taured', memmap='w+',   shape=delta.shape)

    Dplus = A.cosmology.Dplus
    FOmega  = A.cosmology.FOmega
    Dc = A.cosmology.Dc
    Ea = A.cosmology.Ea
    dir = A.sightlines.dir
   
    with sharedmem.MapReduce() as pool:
        def convolve(i):
            ind = index[i]
            dreal = dc[ind]
            arg = dreal.argsort()
            dreal = dreal[arg]
            ind = ind[arg]
            dr = dir[i]
            #with pool.critical:
            losdisp = disp[ind]
            a = Dc.inv(dreal)
            Dfactor = Dplus(a) / Dplus(1.0)
            deltaLN = numpy.exp(Dfactor * delta[ind] - (Dfactor ** 2 * var) * 0.5)
            T = A.IGMTemperature * (1 + deltaLN) ** (5. / 3. - 1)
            vtherm = SQRT_KELVIN_TO_KMS * T ** 0.5
            rsd = losdisp * FOmega(a) * Dfactor / A.DH
            dred = dreal + rsd
            dtherm = vtherm / A.C / (a * Ea(a))
#            dtherm[:] = 0
#            dred[:] = dreal
            taureal[ind] = numpy.float32(deltaLN)
            taured[ind] = numpy.float32(irconvolve(dreal, dred, taureal[ind],
                dtherm))
            #print irconvolve(dreal, dred, taureal[ind], dtherm)
            print i, len(ind), dtherm.max() / (dreal[1] - dreal[0])
            assert not numpy.isnan(taured[ind]).any()
#            taureal[ind] = numpy.float32(irconvolve(dreal, dreal, taureal[ind],
#                dtherm))
            Zqso = qsocatelog[i]['Z_REAL']
            dqso = Dc(1 / (Zqso + 1.0))
            dqso = dqso + numpy.interp(dqso, dreal, rsd)
            qsocatelog[i]['Z_VI'] = 1.0 / Dc.inv(dqso) - 1
            #with pool.critical:
        pool.map(convolve, range(len(A.sightlines)))

    taured.flush()
    taureal.flush()
    qsocatelog.flush()
class IndexBySightline(object):
    def __init__(self, A):
        objectid = A.P('objectid')
        N = len(A.sightlines)
        ind = sharedmem.argsort(objectid)
        sorted = objectid[ind]
        self.start = sharedmem.searchsorted(sorted, 
                numpy.arange(N), side='left')
        self.end = sharedmem.searchsorted(sorted, 
                numpy.arange(N), side='right')
        self.ind = ind
    def __getitem__(self, i):
        return self.ind[self.start[i]:self.end[i]]
