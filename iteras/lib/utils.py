import re
import time
import types
import typing
import importlib

from iteras import registry
from iteras import xp_fft


def camelcase_to_snakecase(s):
    '''
    Convert CamelCase to snake_case.
    Underscores are not inserted in case of acronyms (like CCD)
    or when the uppercase letter is preceded by a number like M2C.
    '''
    tokens = re.findall(r'[A-Z]{3,}(?=[A-Z][a-z])|[A-Z]+[0-9a-z]*', s)
    result = [tokens[0]]
    for t in tokens[1:]:
        if not result[-1][-1].isdigit():
            result.append('_')
        result.append(t)
    return ''.join(x.lower() for x in result)


class AmbiguousClassError(ImportError):
    '''
    Raised by import_class() when a class name corresponds to different
    classes in more than one package.
    '''


def _import_from_package(package, classname, modulename):
    '''
    Return the class *classname* from module *package.modulename*,
    or None if that module does not exist.
    Raises AttributeError if the module exists but does not define the class.
    '''
    module_to_import = f'{package}.{modulename}'
    try:
        mod = importlib.import_module(module_to_import)
    except ModuleNotFoundError as e:
        match = re.search(r"No module named '([^']+)'", str(e))
        # Check if the failed import was for the wanted module (or one of its
        # parent packages), or something else (e.g. imports of missing
        # third-party packages inside that module)
        if match:
            missing = match.group(1)
            if module_to_import == missing or module_to_import.startswith(missing + '.'):
                return None
        raise
    try:
        return getattr(mod, classname)
    except AttributeError:
        raise AttributeError(f'Class {classname} not found in file {modulename}.py') from None


def _search_packages(packages, classname):
    '''
    Look for *classname* in all *packages*. Returns the class, or None if it
    is not found. Raises AmbiguousClassError if different classes are found.
    The same class found in more than one package (e.g. through a module
    that is an alias of another one) is not ambiguous.
    '''
    modulename = camelcase_to_snakecase(classname)
    found = {}
    for package in packages:
        klass = _import_from_package(package, classname, modulename)
        if klass is not None:
            found.setdefault(klass, []).append(f'{package}.{modulename}')
    if len(found) > 1:
        where = '; '.join(f'{klass.__module__}.{klass.__qualname__} (found in {", ".join(mods)})'
                          for klass, mods in found.items())
        raise AmbiguousClassError(
            f'Class name {classname} is ambiguous, it corresponds to different classes: {where}. '
            f'Use a qualified name to choose one, e.g. "<plugin>.{classname}" or the full '
            f'module path "<package>.<module>.{classname}"')
    return next(iter(found), None)


def _find_class(classname, additional_modules):
    '''import_class() without the automatic loading of plugins'''
    packages = list(dict.fromkeys(registry.search_packages() + additional_modules))
    prefix, _, name = classname.rpartition('.')
    if not prefix:
        return _search_packages(packages, name)

    # Qualified name: look only in the packages of that plugin (or package)
    candidates = [p for p in packages if p == prefix or p.startswith(prefix + '.')]
    if candidates:
        return _search_packages(candidates, name)

    # Otherwise, a full module path
    try:
        mod = importlib.import_module(prefix)
    except ModuleNotFoundError as e:
        if e.name is not None and (prefix == e.name or prefix.startswith(e.name + '.')):
            return None
        raise
    try:
        return getattr(mod, name)
    except AttributeError:
        raise AttributeError(f'Class {name} not found in module {prefix}') from None


def import_class(classname, additional_modules=None):
    '''
    Dynamically import a class by name from the registered packages.

    Given a class name in CamelCase, this function looks for the class
    in all the following packages:
        - iteras.processing_objects
        - iteras.data_objects
        - iteras.display
        - the packages registered by plugins (see iteras.registry)
        - user-defined additional modules, if any

    The module name is inferred by converting the class name from CamelCase to
    snake_case: class FooBar must be defined in a module called foo_bar.py.
    If the class name corresponds to different classes in more than one package,
    AmbiguousClassError is raised.

    The class name can be qualified, to choose one of several classes with the same
    name or to avoid searching all packages:
        - "<prefix>.FooBar", where <prefix> is a plugin (e.g. "specula.FooBar") or
          one of the searched packages: the class is searched only in the packages
          equal to <prefix> or contained in it;
        - otherwise, the full path of the module defining the class, e.g.
          "mypackage.mymodule.FooBar".

    If the class is not found, the plugins declared with an "iteras.plugins" entry
    point are loaded (once per process, see iteras.registry) and the search is
    repeated.

    Parameters
    ----------
    classname : str
        The name of the class to import (in CamelCase), possibly qualified.
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
    AmbiguousClassError
        If the class name corresponds to different classes.
    AttributeError
        If the class is not found in the located module.
    '''
    additional_modules = list(additional_modules or [])

    klass = _find_class(classname, additional_modules)
    if klass is None and registry.load_entry_point_plugins():
        klass = _find_class(classname, additional_modules)
    if klass is None:
        modulename = camelcase_to_snakecase(classname.rpartition('.')[2])
        raise ImportError(f'Class {classname} must be defined in a file called {modulename}.py '
                          'but it cannot be found')
    return klass


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


def make_tn():
    return time.strftime("%Y%m%d_%H%M%S")


def flatten(x):
    '''
    Generator that will flatten a list that may contain
    other lists (nested arbitrarily) and simple items
    into a flat list.

    >>> flat = flatten([[1,[2,3]],4,[5,6]])
    >>> list(flat)
    [1,2,3,4,5,6]

    '''
    for item in x:
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def resolve_type(tp, require_list=False, require_dict=False):
    '''
    Extract type information from compound type declaration:
    List[Recmat] -> Recmat
    Dict[str, Recmat] -> Recmat
    Recmat | None -> Recmat
    '''
    union_types = [typing.Union, types.UnionType]

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if require_list and origin != list:
        raise TypeError
    if require_dict and origin != dict:
        raise TypeError

    if origin == dict:
        typ = args[1]
    elif origin == list:
        typ = args[0]
    elif origin in union_types:
        typ = args[0]
    else:
        typ = tp

    return typ


def psd_to_signal(psd, fs, xp, dtype, complex_dtype, seed=1):
    """
    Generate a random signal with a given PSD and sampling frequency.
    
    Parameters:
    -----------
    psd : 1D array
        Power spectral density (PSD) of the signal
        Note: unit [X^2/Hz], where X is the unit of the signal.
    fs : float
        Sampling frequency (unit [Hz]).
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
    temp = xp_fft(xp).ifft(pspectrum) * 2 * n
    out = xp.real(temp).astype(dtype)
    im = xp.imag(temp).astype(dtype)
    return out, im
