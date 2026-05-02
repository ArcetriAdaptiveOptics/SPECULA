#!/usr/bin/env python3
import argparse
import sys
import os

import specula
specula.init(-1)  # Ensure Specula is initialized before importing data objects
                  # -1 means CPU (we do not need GPU for plotting)
from specula.data_objects.ifunc import IFunc
from specula.data_objects.m2c import M2C
from specula.lib.plot_utils import display_ifunc_2d

def main():
    """
    Main function to load an IFunc FITS file from disk and plot its modes.
    """
    # 1. Setup the argument parser
    parser = argparse.ArgumentParser(
        description="Plot Influence Functions from a FITS file using Specula."
    )

    # Positional argument (mandatory)
    parser.add_argument(
        "filename", 
        type=str,
        help="Path to the IFunc FITS file to be plotted."
    )

    # Optional arguments
    parser.add_argument(
        "--start-mode", 
        type=int,
        default=0,
        help="The ID of the first mode to display (default: 0)."
    )
    parser.add_argument(
        "--cols", 
        type=int,
        default=10,
        help="Number of rows and columns for the grid display (default: 10)."
    )
    parser.add_argument(
        "--no-ticks", 
        action="store_true",
        help="Flag to hide axes ticks in the plot."
    )
    parser.add_argument(
        "--m2c-file",
        type=str,
        default=None,
        help="Optional path to an M2C FITS file. If provided, M2C is applied before plotting."
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # 2. Check if file exists
    if not os.path.isfile(args.filename):
        raise FileNotFoundError(f"The file '{args.filename}' does not exist.")

    try:
        # 3. Load the IFunc object from disk
        print(f"Loading IFunc from: {args.filename}...")
        my_ifunc = IFunc.restore(args.filename)

        my_m2c = None
        if args.m2c_file is not None:
            if not os.path.isfile(args.m2c_file):
                raise FileNotFoundError(f"The M2C file '{args.m2c_file}' does not exist.")
            print(f"Loading M2C from: {args.m2c_file}...")
            my_m2c = M2C.restore(args.m2c_file)

        # 4. Display the IFunc
        print(f"Plotting grid of {args.cols}x{args.cols} modes starting"
              f" from mode {args.start_mode}...")
        display_ifunc_2d(
            ifunc_obj=my_ifunc,
            m2c_obj=my_m2c,
            id_mode_starting=args.start_mode,
            n_raw_col=args.cols,
            do_not_show_ticks=args.no_ticks
        )

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
