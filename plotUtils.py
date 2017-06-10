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

        d = [[self._targetFun(y / samplesCount, x / samplesCount)
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


class PlotHistory(BasePlot):

    def __init__(self, numberOfPlots):
        super().__init__()

        self._plots = [plt.plot([-10.0, 10.0], [-10.0, 10.0])[0]
                       for i in range(numberOfPlots)]
        self._data = [[] for i in range(numberOfPlots)]

        plt.show(False)

    def entry(self, values):
        assert len(values) == len(self._plots)
        assert len(values) == len(self._data)

        for i in range(len(self._data)):
            self._data[i].append(values[i])

    def update(self):
        super().update()

        xData = list(range(len(self._data[0])))
        for i, p in enumerate(self._plots):
            #assert (isinstance(xData[0], int), isinstance(self._data[i][0],float))
            assert len(xData) == len(self._data[i])
            p.set_xdata(xData)
            p.set_ydata(self._data[i])

            ax = plt.gca()
            # recompute the ax.dataLim
            ax.relim()
            # update ax.viewLim using the new dataLim
            ax.autoscale_view()
            plt.draw()

        plt.autoscale(enable=True, axis='both', tight=True)

        plt.pause(0.001)
