#!/usr/bin/env python3
import scipy.optimize as opt
from cec2013lsgo.cec2013 import Benchmark
from des import DEStrategy

bench = Benchmark()


def runDE():
    # TODO: why is it so slow
    for f in range(1, bench.get_num_functions() + 1):
        print('--------------------------------')
        print(f"Running DE on CEC test {f} :")
        info = bench.get_info(f)
        fun = bench.get_function(f)
        print(f"Test info: {info}")

        boundaries = [(info['lower'], info['upper'])] * info['dimension']
        lambd = info['dimension'] // 200
        print(f'Population size {lambd}')

        def callback(best, **kwargs):
            # print("Next iteration: ")
            # print(fun(best))
            print(kwargs)
            # input("press any key to continue...")

        ret = opt.differential_evolution(
            # lambda x: (print("e"), fun(x))[1],
            fun,
            boundaries,
            popsize=lambd,
            callback=callback,
            disp=True, init='latinhypercube')
        print(res)


def runDES():
    for f in range(1, bench.get_num_functions() + 1):
        print('--------------------------------')
        print(f"Running DES on CEC test {f} :")
        info = bench.get_info(f)
        fun = bench.get_function(f)
        print(f"Test info: {info}")

        boundaries = [(info['lower'], info['upper'])] * info['dimension']
        # lambd = info['dimension']
        # print(f'Population size {lambd}')

        for population, avgStddev in DEStrategy(
                n=info['dimension'],
                targetFun=fun,
                boundaries=boundaries,
                F=1 / (2**0.5) - 0.34
                # F = 1/(2**0.5) # TODO: why won't converge
        ):
            print('Current best:', max(map(fun, population)))
            # print('Current population:')
            # for p in population:
            #     print('result:', fun(p))
            print('Current standard deviation:', avgStddev)

if __name__ == '__main__':
    runDE()
    runDES()
