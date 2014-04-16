import sharedmem
def chunkmap(func, list, chunksize, reduce=None):
    N = len(list)
    def work(j):
        return [func(list[i])
            for i in range(j, min([j + chunksize, N]))]
    if reduce is not None:
        def myreduce(rt):
            return [reduce(a) for a in rt]
    else:
        myreduce = None
    with sharedmem.MapReduce() as pool:
        pool.map(work, range(0, N, chunksize), reduce=myreduce)

