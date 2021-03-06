#!/usr/bin/env python3
import scipy.optimize as opt
from des import DEStrategy, randomizePopulation
import multiprocessing
import plotUtils
import numpy as np
from functions import fun1

MAX_ITERATIONS = 500
FUNCTIONS = [
    fun1,
    fun1,
    fun1,
    fun1,
]
DIMENSION = [
    5,
    10,
    15,
    20,
]
BOUNDS = [
    [(0,5)]*d for d in DIMENSION
]
FUNCTIONS_TO_RUN = [0,1,2,3]
THREADS_COUNT = 4

def runDEOneFunc(funcId):
    fun = FUNCTIONS[funcId]
    boundaries=BOUNDS[funcId]

    result = []
    curIteration = 0
    def callback(best, **kwargs):
        nonlocal curIteration
        result.append(fun(best))
        curIteration += 1
        if curIteration == MAX_ITERATIONS:
            return True
        # print(kwargs)
        # input("press any key to continue...")

    ret = opt.differential_evolution(
        fun,
        boundaries,
        popsize=4,  # 4 is multipler (base is dimension)
        callback=callback,
        strategy='rand1bin',
        # disp=True,
        init='latinhypercube')
    if len(result) < MAX_ITERATIONS:
        result += [result[-1]] * (MAX_ITERATIONS-len(result))
    return result

def runDE():
    print('--------------------------------')
    print(f"Running DE:")
    pool = multiprocessing.Pool(THREADS_COUNT)
    return list(pool.map(runDEOneFunc, FUNCTIONS_TO_RUN))

def runDESOneFunc(funcId):
    fun = FUNCTIONS[funcId]
    boundaries=BOUNDS[funcId]

    result = []
    curIteration = 0
    for best, population, avgStddev in DEStrategy(
            n=DIMENSION[funcId],
            targetFun=fun,
            boundaries=boundaries,
            F=1 / (2**0.5) - 0.03,
    ):
        # print(f'Current best score: {fun(best)}')
        # print('Current standard deviation:', avgStddev)
        result.append(fun(best))
        curIteration += 1
        if curIteration == MAX_ITERATIONS:
            break
    return result

def runDES():
    print('--------------------------------')
    print(f"Running DES:")
    pool = multiprocessing.Pool(THREADS_COUNT)
    return list(pool.map(runDESOneFunc, FUNCTIONS_TO_RUN))

def runRandomSamplingOneFunc(funcId):
    fun = FUNCTIONS[funcId]
    boundaries=BOUNDS[funcId]

    result = []
    best = float("inf")
    for curIteration in range(MAX_ITERATIONS):
        population = randomizePopulation(4*DIMENSION[funcId], boundaries)
        best = min(best, min(map(fun, population)))
        result.append(best)
    return result

def runRandomSampling():
    print('--------------------------------')
    print(f"Running runRandomSampling:")
    pool = multiprocessing.Pool(THREADS_COUNT)
    return list(pool.map(runRandomSamplingOneFunc, FUNCTIONS_TO_RUN))

if __name__ == '__main__':
    #results = (runDE(),runDES())
    results = (runDES(), runDE(), runRandomSampling())
    for r in results:
        for c in r:
            print(c)
        print()
        print()

    plots = []
    for res in zip(*results):
        assert(len(res) == len(results))
        p = plotUtils.PlotHistory(len(res))
        p.update()
        for v in zip(*res):
            assert(len(v) == len(results))
            p.entry(v)
            p.update()
        p.update()
        plots.append(p)
    input()
    for i, p in enumerate(plots):
        p.save(f"fun1_{str(i+1)}.eps")
        p.save(f"fun1_{str(i+1)}.svg")
        p.save(f"fun1_{str(i+1)}.png")
    input()
