from mpi4py import MPI
import numpy
import sharedmem

world = MPI.COMM_WORLD

def cov(arr, rowvar=0, ddof=0, world=MPI.COMM_WORLD):
    """ 
        arr: input array, arr is divided in Nsample direction.
           if rowvar is 0, arr is (Nsample, Nvar) 
           if rowvar is not 0 , arr is (Nvar, Nsample) 
        normalization is Nsample - ddof 

        returns (Nvar, Nvar) covariance matrix on all ranks
    """
    if rowvar:
        myarr = arr.T
    else:
        myarr = arr

    myarr = numpy.atleast_2d(myarr)

    # find the true Nvar. some myarr can be shape(0, 0)
    Nsample, Nvar = myarr.shape
    Nvar = world.allreduce(Nvar, op=MPI.MAX)

    myarr = myarr.reshape(-1, Nvar)
    Nsample, Nvar = myarr.shape

    # total number of samples
    Nsample = world.allreduce(Nsample)

    print 'Nsample = ', Nsample
    mysum = myarr.sum(axis=0)

    # total sum/
    sum = world.allreduce(mysum)

    # cross terms on me
    mycross = numpy.empty((Nvar, Nvar))
    with sharedmem.MapReduce() as pool:
        def work(i):
            tmp = numpy.einsum(
            "ij,i->j", myarr, myarr[:, i])
            return i, tmp
        def reduce(i, tmp):
            mycross[i] = tmp
        pool.map(work, range(Nvar), reduce=reduce)

    # total of cross terms
    cross = world.allreduce(mycross)
    mean = sum / (Nsample - ddof)
    if False:
        if world.rank == 0:
            print 'reduced cov', cross
        if world.rank == 0:
            print 'reduced mean', mean, Nsample, ddof
    cov = cross / (Nsample - ddof) - mean[:, None] * mean[None, :]
    if world.rank == 0:
        print Nsample, ddof
    return cov

def test():
    world = MPI.COMM_WORLD
    numpy.random.seed(100)
    arr = numpy.random.random(size=(1200, 2000)) + \
            numpy.sin(numpy.arange(2000) * 0.4)[None, :] * 0.3 *\
            numpy.arange(1200)[:, None]
    arr[:, 0] = numpy.nan
    print numpy.array_split(arr, world.size)

    myarr = world.scatter(numpy.array_split(arr, world.size))
    result = cov(myarr)
    if world.rank == 0:
        result2 = numpy.cov(arr, rowvar=0, ddof=0)
        print 'true cross', numpy.einsum('ij,ik->jk', arr, arr)
        print 'true mean', arr.mean(axis=0)
        print 'true cov', 1 / 2. * numpy.einsum('ij,ik->jk', arr, arr) - \
        arr.mean(axis=0)[:, None] * arr.mean(axis=0)[None, :]
        print 'par', numpy.diag(result)
        print 'seq', numpy.diag(result2)
        print (result - result2).max()
