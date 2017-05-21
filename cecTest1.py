#!/usr/bin/env python
from cec2013lsgo.cec2013 import Benchmark

bench = Benchmark()

for f in range(1,bench.get_num_functions()+1):
    print(bench.get_info(f))
    # print(bench.get_function(f))
