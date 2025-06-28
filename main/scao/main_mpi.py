# mpiexec -n 2 python script.py args

from mpi4py import MPI
from mpi4py.util import pkl5

import sys

import cProfile
from pstats import Stats

#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--overrides', type=str)
parser.add_argument('--target', type=int, default=0)
parser.add_argument('yml_file', nargs='+', type=str, help='YAML parameter files')
parser.add_argument('--diagram', action='store_true', help='Save image block diagram')
parser.add_argument('--diagram-title', type=str, default=None, help='Block diagram title')
parser.add_argument('--diagram-filename', type=str, default=None, help='Block diagram filename')

if __name__ == '__main__':
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    args = parser.parse_args()

    N = 100000000
    datatype = MPI.FLOAT
    num_bytes = N * (datatype.Pack_size(count=1, comm=comm) + MPI.BSEND_OVERHEAD)

    attached_buf = bytearray(num_bytes)
    MPI.Attach_buffer(attached_buf)

    print('Starting proceess with rank:', rank)
    if args.cpu:
        target_device_idx = -1
    else:
        target_device_idx = args.target

    import specula

    mpidbg = True
    specula.init(target_device_idx, precision=1, rank=rank, comm=comm, mpi_dbg=mpidbg)

    print(args)    
    from specula.simul import Simul
    simul = Simul(*args.yml_file,
                  overrides=args.overrides,
                  diagram=args.diagram,
                  diagram_filename=args.diagram_filename,
                  diagram_title=args.diagram_title  
    )
    simul.run()

    MPI.Detach_buffer()


'''
def addRankToIniName(name, r):
    name_no_ext, ext = name.split('.')
    return name_no_ext+str(r)+'.'+ext

def main(*inifiles):
    global comm
    global rank
    param_files = list(inifiles)
    param_files[0] = addRankToIniName(param_files[0], rank)
    #print(param_files)
    simul = Simul(*param_files)
    simul.run()

if __name__ == '__main__':
    if sys.argv[3]=='profile':
        with cProfile.Profile() as pr:
            main(*sys.argv[4:])
        stats = Stats(pr).sort_stats("cumtime")
        stats.print_stats(r"\((?!\_).*\)$", 200)
    else:
        main(*sys.argv[3:])
'''