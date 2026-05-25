import importlib
import inspect
import pkgutil
from astropy.io import fits
from collections.abc import Mapping, Sequence, Set

from specula import cp, np
import specula.data_objects
import specula.processing_objects

def cpu_and_gpu(f):
    '''
    Decorator to run a test method first on GPU (if available)
    and the on CPU. If the GPU is not available, it will be
    skipped silently.
    '''
    def test_gpu(self):
        return f(self, target_device_idx=0, xp=cp)
    
    def test_cpu(self):
        return f(self, target_device_idx=-1, xp=np)
    
    def test_both(self):
        if cp is not None:
            test_gpu(self)
        test_cpu(self)
        
    return test_both

def cpu_and_gpu_noself(f):
    '''
    Decorator to run a test function first on GPU (if available)
    and the on CPU. If the GPU is not available, it will be
    skipped silently.
    '''
    def test_gpu():
        return f(xp=cp)
    
    def test_cpu():
        return f(xp=np)
    
    def test_both():
        if cp is not None:
            test_gpu()
        test_cpu()
        
    return test_both

def assert_HDU_contents_match(data_path, ref_path, decimal=5):
    '''
    Assert that the data contents of two FITS file are almost equal
    up to a certain number of decimals (default 5).

    Both FITS files will be opened and examined HDU by HDU using
    np.testing.assert_array_almost_equal
    '''
    with fits.open(data_path) as data:
        with fits.open(ref_path) as ref:
            for i, (gen_hdu, ref_hdu) in enumerate(zip(data, ref)):
                if hasattr(gen_hdu, 'data') and hasattr(ref_hdu, 'data') and gen_hdu.data is not None:
                    np.testing.assert_array_almost_equal(
                        gen_hdu.data, ref_hdu.data,
                        decimal=decimal,
                        err_msg=f"Data in HDU #{i} does not match reference"
                    )


def iter_data_object_classes(skip=None, require_methods=None):
    """
    Iterate over classes defined in ``specula.data_objects`` submodules.

    Parameters
    ----------
    skip : iterable of str, optional
        Class names to exclude.
    require_methods : iterable of str, optional
        If provided, only classes exposing all listed attributes are yielded.
    """
    skip = set(skip or [])
    required = tuple(require_methods or [])

    for _, module_name, _ in pkgutil.iter_modules(specula.data_objects.__path__):
        full_name = f"{specula.data_objects.__name__}.{module_name}"
        module = importlib.import_module(full_name)

        for class_name, klass in inspect.getmembers(module, inspect.isclass):
            if class_name in skip:
                continue
            if klass.__module__ != module.__name__:
                continue
            if class_name.startswith('_'):
                continue
            if required and not all(hasattr(klass, meth) for meth in required):
                continue

            yield klass


def iter_processing_object_classes(skip=None):
    """
    Iterate over classes defined in ``specula.processing_object`` submodules.

    Parameters
    ----------
    skip : iterable of str, optional
        Class names to exclude.
    """
    skip = set(skip or [])

    for _, module_name, _ in pkgutil.iter_modules(specula.processing_objects.__path__):
        full_name = f"{specula.processing_objects.__name__}.{module_name}"
        module = importlib.import_module(full_name)

        for class_name, klass in inspect.getmembers(module, inspect.isclass):
            if class_name in skip:
                continue
            if klass.__module__ != module.__name__:
                continue
            if class_name.startswith('_'):
                continue

            yield klass


def find_instances(obj, cls, *, seen=None, path="root"):
    """
    Recursively search an object graph for instances of `cls`.

    Args:
        obj: The object to inspect.
        cls: The target class/type to search for.
        seen: Internal set of visited object ids (prevents infinite recursion).
        path: Internal path string showing where the object was found.

    Returns:
        A list of tuples: (path, matching_object)
    """

    excluded_names = ['xp', 'np']
    excluded_types = ["<class 'specula.data_objects.simul_params.SimulParams'>"]

    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    # Match
    if isinstance(obj, cls):
        yield (path, obj)

    # Dict-like
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            yield from find_instances(value, cls, seen=seen, path=f"{path}[{key!r}]")

    # List/tuple/set-like (but not strings/bytes)
    elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, (str, bytes, bytearray)):
        for i, item in enumerate(obj):
            yield from find_instances(item, cls, seen=seen, path=f"{path}[{i}]")

    # Regular objects with attributes
    elif hasattr(obj, "__dict__"):
        for attr_name, value in vars(obj).items():
            if (attr_name not in excluded_names) and (str(type(value)) not in excluded_types):
                yield from find_instances(value, cls, seen=seen, path=f"{path}.{attr_name}")

