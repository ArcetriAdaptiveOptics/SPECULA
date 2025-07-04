import sys

import specula

specula.init(device_idx=int(sys.argv[1]), precision=int(sys.argv[2]))

from specula.simul import Simul

import cProfile
from pstats import Stats

def main(*inifiles):
    simul = Simul(*inifiles)
    simul.run()

if __name__ == '__main__':
    if sys.argv[3]=='profile':
        with cProfile.Profile() as pr:
            main(*sys.argv[4:])
        stats = Stats(pr).sort_stats("cumtime")
        stats.print_stats(r"\((?!\_).*\)$", 200)
    else:
        main(*sys.argv[3:])
