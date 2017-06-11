from random import uniform, gauss
import numpy as np
from math import sqrt, pi
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
    ret = sum(abs(p - avgPoint) for p in population) / len(population)
    ret = sum(np.sqrt(ret)) / len(ret)
    return ret


def averagePoint(population):
    return sum(population) / len(population)


def inBounds(vec, bounds):
    assert(len(vec) == len(bounds))
    return all(down <= co <= up for co, (down, up) in zip(vec, bounds))


def DEStrategy(
        n,
        targetFun,
        boundaries,

        lambd=None,
        mu=None,
        F=1 / (2**0.5),
        c=None,
        H=None,
        e=None,
        initialPopulation=None,
):
    """Finds the global minimum of a multivariate function.

    Parameters
    ----------
    n : int
        Problem dimensions.
    targetFun : callable
        The objective function to be minimized.  Must be in the form
        ``f(x)``, where ``x`` is the argument in the form of a 1-D array.
    lambd : int, optional
        Quantity of population.  default 4 * n.
    mu : int, optional
        Quantity of best individuals selection.  Default lamd//2.
    F : int, optional
        Scaling factor. Dafault 1 / sqrt(2).
    c : int, optional
        Midpoint smootching factor. Default 4 / (n + 4)
    H : int, optional
        Time horizon for population archive. Default 6 + int(3 * sqrt(n))
    e : int, optional
        Noise intensity. Default 10 ** (-8) / sqrt(n).
    initialPopulation : sequence, optional
        List of lambd individuals.
    boundaries : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x) == n``.
    """

    smallDelta = sqrt(2/pi) * sqrt(n)

    if lambd is None:
        lambd = 4 * n
    assert(isinstance(lambd, int))

    if mu is None:
        mu = lambd // 2
    assert(isinstance(mu, int))

    if c is None:
        c = 4 / (n + 4)
    assert(isinstance(c, float))

    if H is None:
        H = 6 + int(3 * sqrt(n))
    assert(isinstance(H, int))

    if e is None:
        e = 10e-8 / smallDelta
    assert(isinstance(e, float))

    if initialPopulation is None:
        initialPopulation = randomizePopulation(lambd, boundaries)

    def penaltyFun(x, qmax):
        anyOutside = False
        ret = 0
        for coord, (down, up) in zip(x, boundaries):
            if coord > up:
                anyOutside = True
                ret += (coord - up)**2
            if coord < down:
                anyOutside = True
                ret += (down - coord)**2
        if anyOutside:
            ret += qmax
            ret -= targetFun(x)
        return ret

    def targetWithPenaltyFun(x, qmax):
        return targetFun(x) + penaltyFun(x, qmax)

    iterationNumber = 0
    delta = 0
    population = initialPopulation
    population.sort(key=lambda x: targetFun(x))
    bestTillNow = population[0]
    yield bestTillNow, population, estimateStdev(population, averagePoint(population))
    populationHistory = [population]

    qmax = -float("inf")
    qmax = max(qmax, max(map(targetFun, population)))
    while True:
        avgPoint = averagePoint(population)

        population.sort(key=lambda x: targetWithPenaltyFun(x, qmax))

        muPopulation = population[0:mu]
        muAvgPoint = averagePoint(muPopulation)

        delta = (1 - c) * delta + c * (muAvgPoint - avgPoint)

        newPopulation = []
        for i in range(lambd):
            oldPop = random.choice(populationHistory)
            ind1, ind2 = [random.choice(oldPop) for _ in range(2)]
            d = F * (ind1 - ind2) + delta * smallDelta * random.gauss(0, 1)
            newInd = muAvgPoint + d + e * np.random.normal(0, 1, n)
            qmax = max(qmax, targetFun(newInd))
            if targetWithPenaltyFun(newInd, qmax) < targetWithPenaltyFun(bestTillNow, qmax):
                assert(penaltyFun(bestTillNow, qmax) == 0)
                assert(inBounds(bestTillNow, boundaries))
                assert(inBounds(newInd, boundaries))
                bestTillNow = newInd
            newPopulation.append(newInd)

        population = newPopulation.copy()
        yield bestTillNow, population, estimateStdev(population, averagePoint(population))
        populationHistory.append(population)

        if (len(populationHistory) > H):
            populationHistory = populationHistory[1:]
