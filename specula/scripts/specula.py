#!/usr/bin/env python

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsimul', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--overrides', type=str)
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--mpidbg', action='store_true')
    parser.add_argument('--diagram', action='store_true', help='Save image block diagram')
    parser.add_argument('--diagram-title', type=str, default=None, help='Block diagram title')
    parser.add_argument('--diagram-filename', type=str, default=None, help='Block diagram filename')
    parser.add_argument('--diagram-colors-on', action='store_true')
    parser.add_argument('yml_files', nargs='+', type=str, help='YAML parameter files')
    
    args = parser.parse_args()

    import specula
    specula.main_simul(**vars(args))

if __name__ == '__main__':
    main()
