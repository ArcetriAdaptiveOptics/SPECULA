
import re
import typing
import importlib


def camelcase_to_snakecase(s):
    tokens = re.findall('[A-Z]+[0-9a-z]*', s)
    return '_'.join([x.lower() for x in tokens])


def import_class(classname):
    modulename = camelcase_to_snakecase(classname)
    try:
        try:
            mod = importlib.import_module(f'specula.processing_objects.{modulename}')
        except ModuleNotFoundError:
            try:
                mod = importlib.import_module(f'specula.data_objects.{modulename}')
            except ModuleNotFoundError:
                mod = importlib.import_module(f'specula.display.{modulename}')
    except ModuleNotFoundError:
        raise ImportError(f'Class {classname} must be defined in a file called {modulename}.py but it cannot be found')
    
    try:
        return getattr(mod, classname)
    except AttributeError:
        raise AttributeError(f'Class {classname} not found in file {modulename}.py')


def get_type_hints(type):
    hints ={}
    for x in type.__mro__:
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