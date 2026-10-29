#!/usr/bin/env python

import specula
from iteras.scripts.iteras_main import build_parser


def main():
    '''
    Main entry point for command-line specula execution.
    '''
    args = build_parser().parse_args()
    specula.main_simul(**vars(args))

if __name__ == '__main__':
    main()
