
def factors(n, gaps=[1,2,2,4,2,4,2,4,6,2,6], length=11, cycle=3):
    """ fast factorize small integers, based on 
    wheel factorization, which uses a
    cyclic set of gaps between potential primes to greatly reduce the number of
    trial divisions. Here we use a 2,3,5-wheel:
        retrieved from stackoverflow:
           http://stackoverflow.com/a/17000452
    """
    f, fs, next = 2, [], 0
    while f * f <= n:
        while n % f == 0:
            fs.append(f)
            n /= f
        f += gaps[next]
        next += 1
        if next == length:
            next = cycle
    if n > 1: fs.append(n)
    return fs

import numpy
def optimize_size_for_fftw(n, maxfactor=11):
    """ 
        optimize mesh size for fftw
        find a mesh size that is close to n but made of smaller prime factors

    """
    def badness(n1, n):
        f = factors(n1)
        if max(f) > maxfactor: return 999999
        return numpy.abs(n1 - n)

    b = [(badness(n1, n), n1) 
        for n1 in range(int(n * 0.9), int(n * 1.1))]
    b = sorted(b)
    return b[0][1]
        
