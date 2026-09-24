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
    parser.add_argument('--no-speed-report', action='store_true',
                        help='Disable speed report on standard output')
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, MPI_DBG, MPI_SEND_DBG')
    parser.add_argument('--trace-file', type=str, default=None,
                        help='Write per-object timings of each phase (inputs, prepare_trigger, trigger, '
                             'post_trigger, send_outputs) to this tab-separated file, plus a summary '
                             'sorted by total time. With MPI, one file per rank is written.')
    parser.add_argument('--trace-sync', action='store_true',
                        help='With --trace-file, synchronize the GPU at the end of each phase, so that '
                             'timings include GPU execution time (slows down the simulation)')
    parser.add_argument('--trace-skip', type=int, default=0, metavar='N',
                        help='With --trace-file, do not record the first N loop iterations, which include '
                             'one-time costs (FFT plans, kernel compilation, ...). Setup is always recorded.')
    parser.add_argument('--trace-gpu-events', action='store_true',
                        help='With --trace-file, also measure the GPU time of each trigger with CUDA '
                             'events, written as a "trigger_gpu" phase. Lighter than --trace-sync.')
    parser.add_argument('yml_files', nargs='+', type=str,
                        help='YAML parameter files')

    args = parser.parse_args()

    specula.main_simul(**vars(args))

if __name__ == '__main__':
    main()
