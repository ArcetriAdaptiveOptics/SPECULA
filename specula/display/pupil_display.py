import numpy as np

import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def remap_signals(sx_or_sy, slopemap, slope_side_x, slope_side_y):
    """
    Remap the signals based on the slopemap.
    
    Parameters:
        sx_or_sy (numpy.ndarray): Signal vector (sx or sy).
        slopemap (numpy.ndarray): Slopes remapping vector.
        slope_side_x (int): Width of the remapped signal.
        slope_side_y (int): Height of the remapped signal.
    
    Returns:
        numpy.ndarray: Remapped signal.
    """
    # Reshape and remap the signal based on the slopemap.
    remapped_signal = np.zeros((slope_side_x, slope_side_y))
    remapped_signal.flat[slopemap.ravel()] = sx_or_sy
    return remapped_signal


def pupil_display(frame, sx, sy, slopemap, real_ccd_side, cirlceDisp=None, NEGATIVE=False,
                  NOFRAME=False, GET_SIZE=False, TARGET=None, NOSHOW=False, 
                  MAGNIFY=1, slope_side=None, title='pupil_display', do_image_show=False, plot_title=''):
    """
    Display a CCD frame and the corresponding signals.
    
    Parameters:
        frame (numpy.ndarray): CCD frame containing the four sub-pupils.
        sx (numpy.ndarray): x-signal vector.
        sy (numpy.ndarray): y-signal vector.
        slopemap (numpy.ndarray): Slopes remapping vector.
        real_ccd_side (int): CCD frame side (assuming square dimensions).
        cirlceDisp (int): Number of sub-apertures for circle display.
        NEGATIVE (bool): Extend display range to the frame's minimum value.
        NOFRAME (bool): Show only signals without the frame.
        GET_SIZE (bool): Return the display window size.
        TARGET (matplotlib Axes): Target axes for plotting.
        NOSHOW (bool): If True, do not show anything.
        MAGNIFY (int): Magnification factor for the display.
        slope_side (int): Side length for remapping signals.
        title (str): Title for the display window.
        do_image_show (bool): If True, use image_show display method.
        plot_title (str): Title for the plot.
    
    Returns:
        frame_signalx, frame_signaly, GET_SIZE: Remapped signals and window size (if requested).
    """
    
    # Initialize or calculate some parameters
    if slope_side is None:
        slope_side = real_ccd_side
    
    dx = 0.5 if NOFRAME else 1.5
    GET_SIZE = [int(dx * real_ccd_side * MAGNIFY), int(real_ccd_side * MAGNIFY)]
    
    # Exit if NOSHOW is True
    if NOSHOW:
        return
    
    # Prepare the display size
    if do_image_show:
        if NOFRAME:
            xsize, ysize = 640 * MAGNIFY, 480 * MAGNIFY
        else:
            xsize, ysize = 800 * MAGNIFY, 400 * MAGNIFY
    else:
        xsize, ysize = dx * real_ccd_side * MAGNIFY, real_ccd_side * MAGNIFY
    
    if TARGET is None:
        pupil_display.fig, pupil_display.ax = plt.subplots(figsize=(xsize/100, ysize/100))
    else:
        pupil_display.fig, pupil_display.ax = TARGET
    ax = pupil_display.ax
    
    # Display the frame and signals
    if not NOFRAME:
        if do_image_show:
            ax.imshow(frame, cmap='gray')
            ax.set_title(plot_title)
        else:
            if NEGATIVE:
                display_frame = zoom(np.clip(frame, np.min(frame), np.max(frame)), MAGNIFY, order=0)
            else:
                display_frame = zoom(np.clip(frame, 0, np.max(frame)), MAGNIFY, order=0)
            
            ax.imshow(display_frame, cmap='gray')
            ax.set_title(title)
            
            # Circle display if required
            if cirlceDisp:
                for i in range(cirlceDisp):
                    for j in range(cirlceDisp):
                        circ = plt.Circle(((1 + 2 * i) / (2 * cirlceDisp) * real_ccd_side * MAGNIFY,
                                           (1 + 2 * j) / (2 * cirlceDisp) * real_ccd_side * MAGNIFY),
                                          10, color='red', fill=False)
                        ax.add_patch(circ)
    
    # Remap the signals
    if sx.size == slopemap.size:
        frame_signalx = remap_signals(sx, slopemap, slope_side, slope_side)
        frame_signaly = remap_signals(sy, slopemap, slope_side, slope_side)
    else:
        frame_signalx = remap_signals(np.zeros_like(slopemap), slopemap, slope_side, slope_side)
        frame_signaly = remap_signals(np.zeros_like(slopemap), slopemap, slope_side, slope_side)
    
    # Combine and display the signal
    dx = frame_signalx.shape[0]
    dy = frame_signalx.shape[1]
    signal_total = np.zeros((dx//2, dy))
    signal_total[:, :dy//2] = frame_signalx[:dx//2, :dy//2]
    signal_total[:, dy//2:] = frame_signaly[:dx//2, :dy//2]
    
    if do_image_show:
        if NOFRAME or signal_total.shape[1] != frame.shape[1]:
            ax.imshow(signal_total, cmap='gray')
        else:
            combined = np.vstack((frame/np.max(frame) * np.max(signal_total), signal_total))
            ax.imshow(combined, cmap='gray')
    else:
        signal_display = zoom(signal_total, [MAGNIFY/2, MAGNIFY], order=0)
        ax.imshow(signal_display, cmap='gray', vmin=-np.max(np.abs(signal_total)), vmax=np.max(np.abs(signal_total)))
    
    # Show the plot if TARGET is None (meaning we created a new figure)
    if TARGET is None:
        plt.show()
    
    return frame_signalx, frame_signaly, GET_SIZE if GET_SIZE else None, pupil_display.fig, pupil_display.ax
pupil_display.fig = None
pupil_display.ax = None
