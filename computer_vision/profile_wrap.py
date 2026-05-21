import cProfile
import io
import pstats
from pstats import SortKey
import numpy as np
import time

def profilefunc(func, num_of_tests, *args, **kwargs):
    # run once to compile
    func(*args, **kwargs)
        
    pr = start_profiler()
    start_time = time.time()

    for i in np.arange(num_of_tests):
        func(*args, **kwargs)

    end_time = time.time()
    print(end_time - start_time)

    end_profiler(pr)

def start_profiler():
    pr = cProfile.Profile()
    pr.enable()

    return pr

def end_profiler(pr):
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
