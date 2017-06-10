#!/usr/bin/env python
from des import DEStrategy
import scipy.optimize as opt
from plotUtils import Fun2DPlot

from math import sin, cos
import argparse


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Example 2D function minimum optimization problem.')

    parser.add_argument('-a', '--algorithm', help='Use DES or DE.',
                        dest='algorithm', type=str, default='DES')

    return parser.parse_args()


def des(plot, targetFun, boundaries):
    for best, population, avgStddev in DEStrategy(
            n=2,
            targetFun=lambda x: targetFun(*x),
            boundaries=boundaries,
            F=1 / (2**0.5) - 0.04
            # for 2 dimensions F=1/sqrt(2) won't cause converge (but it
            # should?)
    ):
        print(f"Best: {best} Score: {targetFun(*best)}")
        print('Current population:')
        for p in population:
            print(p, 'result:', targetFun(*p))
        print('Current standard deviation:', avgStddev)

        #plot.update(population)
        plot.update([best] + population)
        input()


def de(plot, targetFun, boundaries):
    def callback(*args, **kwargs):
        # TODO: check if it is possible to display whole population
        print('best individual: ', args)
        print(kwargs)
        plot.update([args[0]])
        input()

    opt.differential_evolution(
        lambda x: targetFun(*x),
        boundaries, popsize=8, callback=callback, disp=True, init='latinhypercube')

if __name__ == '__main__':
    args = parseArguments()

    def targetFun(x, y):
        return (sin(x + x**2) + sin(y + x + 1)) * x * (5 - x) * y * (5 - y)

    boundaries = [(0, 5), (0, 5)]

    def normalizedFun(*args):
        return targetFun(*(a + arg * (b - a) for arg, (a, b) in zip(args, boundaries)))

    plot = Fun2DPlot(normalizedFun)
    if args.algorithm == 'DES':
        des(plot, targetFun, boundaries)
    elif args.algorithm == 'DE':
        de(plot, targetFun, boundaries)
    else:
        print('unknown algorithm')
