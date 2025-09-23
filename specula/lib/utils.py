
import re
import typing
import importlib
import warnings

def camelcase_to_snakecase(s):
    '''
    Convert CamelCase to snake_case.
    Underscores are not inserted in case of acronyms (like CCD)
    or when the uppercase letter is preceded by a number like M2C.
    '''
    tokens = re.findall('[A-Z]+[0-9a-z]*', s)
    result = [tokens[0]]
    for i, t in enumerate(tokens[1:]):
        if not result[-1][-1].isdigit():
            result.append('_')
        result.append(t)
    return ''.join([x.lower() for x in result])


def import_class(classname, additional_modules=[]):
    '''
    Dynamically import a class by name from the appropriate specula submodule.

    Given a class name in CamelCase, this function attempts to import the class
    from one of the following submodules (in order):
        - specula.processing_objects
        - specula.data_objects
        - specula.display
        - user-defined additional modules, if any

    The module name is inferred by converting the class name from CamelCase to snake_case.

    Parameters
    ----------
    classname : str
        The name of the class to import (in CamelCase).
    additional_modules: list[str]
        List of additional module names to try

    Returns
    -------
    type
        The class object corresponding to `classname`.

    Raises
    ------
    ImportError
        If the module containing the class cannot be found.
    AttributeError
        If the class is not found in the located module.
    '''
    modulename = camelcase_to_snakecase(classname)
    module_paths = ['specula.processing_objects',
                    'specula.data_objects',
                    'specula.display'] + additional_modules
    
    for module_path in module_paths:
        module_to_import = f'{module_path}.{modulename}'
        try:
            mod = importlib.import_module(module_to_import)
            try:
                return getattr(mod, classname)
            except AttributeError:
                raise AttributeError(f'Class {classname} not found in file {modulename}.py')
        except ModuleNotFoundError as e:
            match = re.search(r"No module named '([^']+)'", str(e))
            # Check if the failed import was for the wanted module,
            # or something else (e.g. imports of missing third-party
            # packages inside that module)
            if match:
                this_module = match.group(1)
                if this_module != module_to_import:
                    # This is not an exception for our module, re-raise it
                    raise

    raise ImportError(f'Class {classname} must be defined in a file called {modulename}.py but it cannot be found')


def get_type_hints(type):
    """
    Collects and returns type hints for the __init__ methods of a class and all its base classes.

    Parameters
    ----------
    type : type
        The class whose __init__ type hints are to be collected.

    Returns
    -------
    dict
        A dictionary mapping parameter names to type hints for all __init__ methods in the class hierarchy.
    """
    hints = {}

    # We iterate on reversed MRO so that child classes will override parent ones.
    for x in reversed(type.__mro__):
        hints.update(typing.get_type_hints(getattr(x, '__init__')))
    return hints

def unravel_index_2d(idxs, shape, xp):
    '''Unravel linear indexes in a 2d-shape (in row-major C order)
    
    Replaces cupy.unravel_index, that forces 2 separate DtH transfers
    '''
    if len(shape) != 2:
        raise ValueError('shape must be 2d')

    idxs = xp.array(idxs).astype(int)
    _, ncols = shape
    row_idx = idxs // ncols
    col_idx = idxs - (row_idx * ncols)
    return row_idx, col_idx

def make_orto_modes(array, xp, dtype):
    """
    Return an orthogonal 2D array
    
    Parameters:
    -----------
    array : 2D array
        Input array
    xp : module
        Array processing module (numpy or cupy)
    dtype : data type, optional
        Data type for arrays
        
    Returns:
    --------
    Q : 2D array
        Orthogonal matrix
    """
    # return an othogonal 2D array
    
    size_array = xp.shape(array)

    if len(size_array) != 2:
        raise ValueError('Error in input data, the input array must have two dimensions.')

    if size_array[1] > size_array[0]:
        Q, R = xp.linalg.qr(array.T)
        Q = Q.T
    else:
        Q, R = xp.linalg.qr(array)

    Q = xp.asarray(Q, dtype=dtype)

    return Q

def is_scalar(x, xp):
    """
    Check if x is a scalar or a 0D array.

    Parameters:
    ----------
    x : object
        The object to check.
    xp : module
        The array processing module (numpy or cupy) to use for checking the shape.
    """
    return xp.isscalar(x) or (hasattr(x, 'shape') and x.shape == ())

def psd_to_signal(psd, fs, xp, dtype, complex_dtype, seed=1):
    """
    Generate a random signal with a given PSD and sampling frequency.
    
    Parameters:
    -----------
    psd : 1D array
        Power spectral density (PSD) of the signal.
    fs : float
        Sampling frequency.
    xp : module
        Array processing module (numpy or cupy) to use for array operations.
    dtype : data type
        Data type for the output signal.
    complex_dtype : data type
        Data type for the complex spectrum.
    seed : int, optional
        Random seed for reproducibility (default is 1).
    """
    n = len(psd)
    df = fs / n / 2.
    rng = xp.random.default_rng(seed)
    # Spectrum vector (complex)
    pspectrum = xp.zeros(2 * n, dtype=complex_dtype)
    # Symmetric spectrum
    pspectrum[1:n+1] = xp.sqrt(psd * df)
    pspectrum[n+1:] = xp.sqrt(psd[:-1] * df)[::-1]
    # Random phase
    ph = rng.uniform(0, 2 * xp.pi, 2 * n)
    pspectrum *= xp.exp(1j * ph, dtype=complex_dtype)
    # Inverse FFT
    temp = xp.fft.ifft(pspectrum) * 2 * n
    out = xp.real(temp).astype(dtype)
    im = xp.imag(temp).astype(dtype)
    return out, im

def local_mean_rebin(arr, mask, xp, block_size=5):
    """
    Compute the local mean over non-overlapping blocks of size block_size x block_size,
    ignoring invalid pixels (where mask is False). Invalid pixels are replaced with NaN,
    so the mean is computed only over valid pixels.

    Parameters:
        arr: 2D array of values (phase, etc.)
        mask: 2D boolean array, True for valid pixels, False for invalid
        xp: numpy or cupy module
        block_size: size of the block for local averaging

    Returns:
        result: 2D array of same shape as arr, filled with local means for invalid pixels
    """
    # Compute global mean only on valid pixels
    global_mean = xp.mean(arr[mask]) if xp.any(mask) else xp.nan

    # Replace invalid pixels with NaN
    arr_valid = xp.where(mask, arr, xp.nan)
    h, w = arr_valid.shape

    # Crop array to be divisible by block_size
    h_crop = h - (h % block_size)
    w_crop = w - (w % block_size)
    arr_valid = arr_valid[:h_crop, :w_crop]

    # Reshape to blocks
    arr_blocks = arr_valid.reshape(h_crop // block_size, block_size, w_crop // block_size, block_size)
    arr_blocks = arr_blocks.transpose(0, 2, 1, 3)  # shape: (h_blocks, w_blocks, block_size, block_size)

    # Compute mean ignoring NaNs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        block_means = xp.nanmean(arr_blocks, axis=(2, 3))
    block_means = xp.where(xp.isnan(block_means), global_mean, block_means)

    # Expand block means back to original shape
    local_mean = xp.repeat(xp.repeat(block_means, block_size, axis=0), block_size, axis=1)

    # Fill result array with local means, keep original shape
    result = xp.full_like(arr, xp.nan, dtype=arr.dtype)
    result[:h_crop, :w_crop] = local_mean

    return result