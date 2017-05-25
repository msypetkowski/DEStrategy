#!/usr/bin/env python
from des import DEStrategy
import scipy.optimize as opt
from plotUtils import Fun2DPlot

from math import sin
import argparse

def parseArguments():
    parser = argparse.ArgumentParser(
        description='Example 2D function minimum optimization problem.')

    parser.add_argument('-a', '--algorithm', help='Use DES or DE.',
                        dest='algorithm', type=str, default='DES')

    return parser.parse_args()

def des(plot, targetFun):
    for population, avgStddev in DEStrategy(
            n=10,
            targetFun=targetFun,
            boundaries=boundaries,
            F=0.1,
            ):
        print('Current population:')
        for p in population:
            print(p, 'result:', targetFun(*p))
        print('Current standard deviation:', avgStddev)

        plot.update(population)
        input()

def de(plot, targetFun):
    def callback(*args, **kwargs):
        # TODO: check if it is possible to display whole population
        print('best individual: ', args)
        print(kwargs)
        plot.update([args[0]])
        input()

    opt.differential_evolution(
            lambda x: targetFun(*x),
            boundaries, popsize=10, callback=callback, disp=True, init='latinhypercube')

if __name__ == '__main__':
    args = parseArguments()

    def targetFun(x,y):
        return (sin(x+x**2) + sin(y+x + 1)) * x * (5-x) * y * (5-y)
    boundaries = [(0, 5), (0, 5)]
    def normalizedFun(*args):
        return targetFun(*(a + arg*(b-a) for arg,(a,b) in zip(args,boundaries)))

    plot = Fun2DPlot(normalizedFun)
    if args.algorithm == 'DES':
        des(plot, targetFun)
    else:
        de(plot, targetFun)
