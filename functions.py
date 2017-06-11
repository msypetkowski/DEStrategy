from itertools import chain
from math import sin, cos
def fun1(vec):
    n = len(vec)
    step = n % 7
    ret = 0
    for i,co1 in enumerate(chain(vec, vec, vec, vec)):
        j = (i*10001 + 127) % 1001
        for co2 in vec[i::step]:
            ret += sin((co1 + co2 + j%17)*(j%7 + 1)) * (j%5 + 1)
        for co2 in vec[i::-step]:
            ret += cos((co1 + co2 + j%13)*(j%5 + 1)) * (j%7 + 2)
    return ret
