#!/usr/bin/env python3
import scipy.optimize as opt
from cec2013lsgo.cec2013 import Benchmark
from des import DEStrategy, randomizePopulation
import multiprocessing
import plotUtils
import numpy as np

bench = Benchmark()

MAX_ITERATIONS = 5
CEC_TO_RUN = [1,2,3,4,5]
THREADS_COUNT = 5

def runDEOneFunc(funcId):
    info = bench.get_info(funcId)
    fun = bench.get_function(funcId)
    print(f"Test info: {info}")

    boundaries = [(info['lower'], info['upper'])] * info['dimension']

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
    print(f"Running DE on CEC tests:")
    pool = multiprocessing.Pool(THREADS_COUNT)
    return list(pool.map(runDEOneFunc, CEC_TO_RUN))

def runDESOneFunc(funcId):
    info = bench.get_info(funcId)
    fun = bench.get_function(funcId)
    print(f"Test info: {info}")

    boundaries = [(info['lower'], info['upper'])] * info['dimension']

    result = []
    curIteration = 0
    for best, population, avgStddev in DEStrategy(
            n=info['dimension'],
            targetFun=fun,
            boundaries=boundaries,
            F=1 / (2**0.5)
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
    print(f"Running DES on CEC tests:")
    pool = multiprocessing.Pool(THREADS_COUNT)
    return list(pool.map(runDESOneFunc, CEC_TO_RUN))

def runRandomSamplingOneFunc(funcId):
    info = bench.get_info(funcId)
    fun = bench.get_function(funcId)
    print(f"Test info: {info}")

    boundaries = [(info['lower'], info['upper'])] * info['dimension']

    result = []
    best = float("inf")
    for curIteration in range(MAX_ITERATIONS):
        population = randomizePopulation(4*info['dimension'], boundaries)
        best = min(best, min(map(fun, population)))
        result.append(best)
    return result

def runRandomSampling():
    print('--------------------------------')
    print(f"Running runRandomSampling on CEC tests:")
    pool = multiprocessing.Pool(THREADS_COUNT)
    return list(pool.map(runRandomSamplingOneFunc, CEC_TO_RUN))

if __name__ == '__main__':
    #results = (runDE(),runDES())
    results = (runDE(), runDES(), runRandomSampling())
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
        p.save(f"cec2013_{str(i+1)}.eps")
        p.save(f"cec2013_{str(i+1)}.svg")
        p.save(f"cec2013_{str(i+1)}.png")
    input()
