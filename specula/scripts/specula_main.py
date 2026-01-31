#!/usr/bin/env python

import argparse
import specula


def main():
    '''
    Main entry point for command-line specula execution.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsimul', type=int, default=1,
                        help='Number of simulations to run')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU execution (equivalent to --target=-1)')
    parser.add_argument('--overrides', type=str,
                        help='YAML string with parameter overrides')
    parser.add_argument('--target', type=int, default=0,
                        help='Target device ID for GPU execution')
    parser.add_argument('--precision', type=int, default=1, choices=[0, 1],
                       help='Floating point precision: 0=double (float64), 1=single (float32)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable python profiler and print stats at the end')
    parser.add_argument('--mpi', action='store_true',
                        help='Use MPI for parallel execution')
    parser.add_argument('--mpidbg', action='store_true',
                        help='Activate MPI debug output')
    parser.add_argument('--stepping', action='store_true',
                        help='Allow simulation stepping')
    parser.add_argument('--diagram', action='store_true',
                        help='Save image block diagram')
    parser.add_argument('--diagram-title', type=str, default=None,
                        help='Block diagram title')
    parser.add_argument('--diagram-filename', type=str, default=None,
                        help='Block diagram filename')
    parser.add_argument('--diagram-colors-on', action='store_true',
                        help='Enable colors in block diagram')
    parser.add_argument('yml_files', nargs='+', type=str,
                        help='YAML parameter files')

    args = parser.parse_args()

    specula.main_simul(**vars(args))

if __name__ == '__main__':
    main()
