#!/usr/bin/env python
from des import DEStrategy
from plotUtils import Fun2DPlot

from math import sin
targetFun = lambda x,y: (sin(x+x**2) + sin(y+x + 1)) * x * (5-x) * y * (5-y)
boundaries = [(0, 5), (0, 5)]

def normalizedFun(*args):
    return targetFun(*(a + arg*(b-a) for arg,(a,b) in zip(args,boundaries)))

plot = Fun2DPlot(normalizedFun)

for population, avgStddev in DEStrategy(
        n=5,
        targetFun=targetFun,
        boundaries=boundaries,
        F=0.1,
        ):
    print("Current population:")
    for p in population:
        print(p, "result:", targetFun(*p))
    print("Current standard deviation:", avgStddev)

    plot.update(population)
    input()
