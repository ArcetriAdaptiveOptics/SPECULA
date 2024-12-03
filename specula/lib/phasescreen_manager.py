import os

from specula import cpuArray
from astropy.io import fits

from specula.lib.calc_phasescreen import calc_phasescreen

def phasescreens_manager(L0, dimension, pixel_pitch, directory, xp, precision, seed=None, verbose=False):
    if seed is None:
        seed = [0]
    
    precision_str = 'single' if precision==1 else 'double'

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # List to store phase screens
    phasescreens = []

    for i, element in enumerate(seed):
        if len(L0) == len(seed):
            L0i = L0[i]
        elif len(L0) == 1:
            L0i = L0[0]
        else:
            raise ValueError("The number of elements in L0 must be 1 or the same as the number of seeds!")

        # Construct the file names
        phasescreen_name = f'ps_seed{xp.around(element)}_dim{xp.around(dimension)}_pixpit{pixel_pitch:.3f}_L0{float(L0i):.4f}_{precision_str}.fits'
        phasescreen_name1 = f'ps_seed{xp.around(element)}_dim{xp.around(dimension)}_pixpit{pixel_pitch:.3f}_L0{xp.around(L0i):.4f}_{precision_str}.fits'
        phasescreen_name2 = f'ps_seed{float(element)}_dim{xp.around(dimension)}_pixpit{pixel_pitch:.3f}_L0{float(L0i):.4f}_{precision_str}.fits'
        phasescreen_name3 = f'ps_seed{float(element)}_dim{xp.around(dimension)}_pixpit{pixel_pitch:.3f}_L0{xp.around(L0i):.4f}_{precision_str}.fits'

        # Check if the phase screen file exists in the directory
        if os.path.exists(os.path.join(directory, phasescreen_name)):
            phasescreen = fits.getdata(os.path.join(directory, phasescreen_name), memmap=True)
        elif os.path.exists(os.path.join(directory, phasescreen_name1)):
            phasescreen = fits.getdata(os.path.join(directory, phasescreen_name1), memmap=True)
        elif os.path.exists(os.path.join(directory, phasescreen_name2)):
            phasescreen = fits.getdata(os.path.join(directory, phasescreen_name2), memmap=True)
        elif os.path.exists(os.path.join(directory, phasescreen_name3)):
            phasescreen = fits.getdata(os.path.join(directory, phasescreen_name3), memmap=True)
        else:
            # Calculate the phase screen if it does not exist
            print('Calculating phasescreen...')
            phasescreen = calc_phasescreen(L0i, dimension, pixel_pitch, seed=element, precision=precision, verbose=verbose, xp=xp)
            fits.writeto(os.path.join(directory, phasescreen_name), cpuArray(phasescreen), overwrite=True)
            print('Done')
        
        # Add the phase screen to the list
        phasescreens.append(phasescreen)

    return phasescreens
