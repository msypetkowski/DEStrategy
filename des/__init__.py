from random import uniform, gauss
import numpy as np
from math import sqrt
import random

def randomizePopulation(lambd, boundaries):
    '''
    :return: new randomized population
    :rtype: list
    '''
    return [np.array([uniform(*b) for b in boundaries])
            for i in range(lambd)]

def estimateStdev(population, avgPoint):
    '''
    Return float with estimated standard deviation of a population.
    '''
    ret = sum(abs(p-avgPoint) for p in population)/len(population)
    ret = sum(np.sqrt(ret)) / len(ret)
    return ret

def averagePoint(population):
    return sum(population)/len(population)

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

    if lambd is None:
        lambd = 4 * n

    if mu is None:
        mu = lambd // 2

    if c is None:
        c = 4 // (n + 4)

    if H is None:
        H = 6 + 3 * sqrt(n)

    if e is None:
        # TODO: make sure it's right
        e = 10 ** (-8) / sqrt(n)

    if initialPopulation is None and boundaries is None:
            raise ValueError( "at least one of parameters:\n"
                    "initialPopulation, boundaries should be not None")

    if initialPopulation is None:
        initialPopulation = randomizePopulation(lambd, boundaries)

    if penaltyFun is None:
        def penaltyFun(x, qmax):
            ret = qmax
            for coord, (down, up) in zip(x, boundaries):
                if coord > up:
                    ret += (coord - up)**2
                if coord < down:
                    ret += (down - coord)**2
            return ret

    iterationNumber = 0
    delta = 0
    population = initialPopulation
    yield population, estimateStdev(population, averagePoint(population))
    populationHistory = [population]

    while True:
        avgPoint = averagePoint(population)

        qmax = max(map(targetFun, population))
        population.sort(key=lambda x: targetFun(x) + penaltyFun(x, qmax))

        muPopulation = population[0:mu]
        muAvgPoint = averagePoint(muPopulation)

        delta = (1-c) * delta + c * (muAvgPoint - avgPoint)

        newPopulation = []
        for i in range(lambd):
            oldPop = random.choice(populationHistory)
            ind1, ind2 = [random.choice(oldPop) for _ in range(2)]
            d = F * (ind1 - ind2) + delta * sqrt(n) * random.gauss(0,1)
            newInd = muAvgPoint + d + e * np.random.normal(0, 1, n)
            newPopulation.append(newInd)

        population = newPopulation.copy()
        yield population, estimateStdev(population, averagePoint(population))
        populationHistory.append(population)

        if (len(populationHistory) > H):
            populationHistory = populationHistory[1:]
