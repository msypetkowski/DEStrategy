#!/usr/bin/env python3
import scipy.optimize as opt
from cec2013lsgo.cec2013 import Benchmark
from des import DEStrategy
import multiprocessing
import plotUtils

bench = Benchmark()

MAX_ITERATIONS = 10

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
    return result

def runDE():
    print('--------------------------------')
    print(f"Running DE on CEC tests:")
    pool = multiprocessing.Pool(4)
    return list(pool.map(runDEOneFunc, [id for id in range(1,5)]))

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
            F=1 / (2**0.5) - 0.34,
            # F = 1/(2**0.5) # TODO: why won't converge
    ):
        print(f'Current best score: {fun(best)}')
        print('Current standard deviation:', avgStddev)
        result.append(fun(best))
        curIteration += 1
        if curIteration == MAX_ITERATIONS:
            break
    return result

def runDES():
    print('--------------------------------')
    print(f"Running DES on CEC tests:")
    pool = multiprocessing.Pool(4)
    return list(pool.map(runDESOneFunc, [id for id in range(1,5)]))

if __name__ == '__main__':
    #results = (runDE(),runDES())
    results = (runDE(), runDES())
    print(results)
    plots = []
    for res in zip(*results):
        assert(len(res) == len(results))
        print("plot lines:", len(res))
        p = plotUtils.PlotHistory(len(res))
        p.update()
        for v in zip(*res):
            assert(len(v) == len(results))
            p.entry(v)
            p.update()
        p.update()
        plots.append(p)
    input()
