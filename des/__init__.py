from random import uniform, gauss
import numpy as np

def randomizePopulation(n, boundaries):
    '''
    :return: new randomized population
    :rtype: list
    '''
    return [np.array([uniform(*b) for b in boundaries])
            for i in range(n)]

def estimateStdev(population, avgPoint):
    '''
    Return float with estimated standard deviation of a population.
    '''
    return sum(abs(p-avgPoint) for p in population)/len(population)

def averagePoint(population):
    return sum(population)/len(population)

# TODO: implement DES
def updatePopulation(population, factor, targetFun):
    '''
    Return population after next iteration
    Performs random mutations.
    Adaptation is n best from 2*n individuals
    (n with and n without mutation)
    '''
    mutated = [p.copy() + np.random.normal(0,factor,len(p))  for p in population]
    ret = population + mutated
    ret.sort(key=lambda x: targetFun(*x))
    return ret[0:len(population)]


def DEStrategy(
        n,
        targetFun,

        lambd=None,
        mu=None,
        F=1/(2**0.5),
        c=None,
        H=None,
        e=None,
        penaltyFun=None,
        initialPopulation=None,
        boundaries=None):

    if initialPopulation is None and boundaries is None:
            raise ValueError( "at least one of parameters:\n"
                    "initialPopulation, boundaries should be not None")

    if initialPopulation is None:
        initialPopulation = randomizePopulation(n, boundaries)

    population = initialPopulation
    while True:
        avgPoint = averagePoint(population)
        estStddev = estimateStdev(population, avgPoint)
        yield population, estStddev
        population = updatePopulation(population, F, targetFun)
