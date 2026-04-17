from astropy.io import fits
from specula import cp, np
import importlib
import inspect
import pkgutil

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
