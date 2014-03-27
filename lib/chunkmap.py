import sharedmem
def chunkmap(func, list, chunksize):
    N = len(list)
    def work(j):
        for i in range(j, min([j + chunksize, N])):
            func(list[i])
    with sharedmem.MapReduce() as pool:
        pool.map(work, range(0, N, chunksize))

