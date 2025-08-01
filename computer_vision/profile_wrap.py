import cProfile
import io
import pstats
from pstats import SortKey

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
