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
    args = parser.parse_args()
    if args.cpu:
        target_device_idx = -1
    else:
        target_device_idx = args.target

    import specula
    specula.init(target_device_idx, precision=1)

    print(args)    
    from specula.simul import Simul
    simul = Simul(*args.yml_file,
                  overrides=args.overrides,
                  diagram=args.diagram,
                  diagram_filename=args.diagram_filename,
                  diagram_title=args.diagram_title,
    )
    simul.run()
