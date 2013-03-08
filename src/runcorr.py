from corr import *
from sys import argv

P = numpy.fromfile('32/all', dtype=pixeldtype)
D = numpy.memmap('out-32/all', mode='r', dtype=datadtype)

tree = buildtree(P)
print 'tree built'
start = int(argv[1])
end = int(argv[2])
print 'runing', start, end
bins = numpy.linspace(0, 200000, 100)
XI = corr(tree, D['delta'], start, end, bins)
numpy.savetxt('XI-%09d-%09d.txt' % (start, end), numpy.asarray(XI).T, fmt=('%g', '%g', '%d'))
print 'done'
