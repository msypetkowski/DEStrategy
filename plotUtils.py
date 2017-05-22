import matplotlib.pyplot as plt
import numpy as np
from itertools import product

currentFigureId = 0

class BasePlot:

    def __init__(self):
        global currentFigureId
        plt.figure(currentFigureId)
        self._figureId = currentFigureId
        currentFigureId += 1

    def update(self):
        plt.figure(self._figureId)

    def __del__(self):
        try:
            plt.figure(self._figureId)
            plt.close()
        except:
            pass

class Fun2DPlot(BasePlot):

    def __init__(self, targetFun):
        super().__init__()

        self._targetFun = targetFun

        self._samplesCount = 128
        samplesCount = self._samplesCount
        d = [[0.0 for y in range(samplesCount)] for x in range(samplesCount)]

        self._p1 = plt.subplot(1, 1, 1)
        self._p1.pcolor(d, cmap=plt.get_cmap('seismic'),
                        vmin=-np.max(np.abs(d)), vmax=np.max(np.abs(d)))

        plt.show(False)

    def update(self, population):
        super().update()
        samplesCount = self._samplesCount

        self._p1.cla()
        samples = [[x / samplesCount, y / samplesCount]
                   for x, y in product(range(samplesCount), range(samplesCount))]

        d = [[self._targetFun(x / samplesCount, y / samplesCount)
              for y in range(samplesCount)] for x in range(samplesCount)]
        maxVal = np.max(np.abs(d))
        minVal = -maxVal
        self._p1.pcolor(d, cmap=plt.get_cmap('seismic'),
                        vmin=minVal, vmax=maxVal)

        xs = []
        ys = []
        population_scaled = list(map(lambda x: x * samplesCount / 5, population))

        for p in population_scaled:
            xs.append(p[0])
            ys.append(p[1])

        self._p1.scatter(xs, ys, s=7, c='g')

        plt.pause(0.001)
