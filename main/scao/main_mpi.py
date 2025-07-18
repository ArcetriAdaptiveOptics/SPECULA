# mpiexec -n 2 python script.py args

try:
    from mpi4py import MPI
    from mpi4py.util import pkl5
    print("mpi4py import successfull. Installed version is:", MPI.Get_version())
except:
    print("mpi4py import failed.")
    print("You should use a main_simul.py for single process execution.")
    exit

import sys

import cProfile
from pstats import Stats

#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--overrides', type=str)
parser.add_argument('--target', type=int, default=0)
parser.add_argument('--mpidbg', action='store_true')
parser.add_argument('yml_file', nargs='+', type=str, help='YAML parameter files')
parser.add_argument('--diagram', action='store_true', help='Save image block diagram')
parser.add_argument('--diagram-title', type=str, default=None, help='Block diagram title')
parser.add_argument('--diagram-filename', type=str, default=None, help='Block diagram filename')

if __name__ == '__main__':
    comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    args = parser.parse_args()

    N = 10000000
    datatype = MPI.FLOAT
    num_bytes = N * (datatype.Pack_size(count=1, comm=comm) + MPI.BSEND_OVERHEAD)

    print(f'MPI buffer size: {num_bytes/1024**2:.2f} MB')
    attached_buf = bytearray(num_bytes)
    MPI.Attach_buffer(attached_buf)

    print('Starting process with rank:', rank)
    if args.cpu:
        target_device_idx = -1
    else:
        target_device_idx = args.target

    import specula

    specula.init(target_device_idx, precision=1, rank=rank, comm=comm, mpi_dbg=args.mpidbg)

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

