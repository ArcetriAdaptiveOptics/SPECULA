import re
import time
import types
import typing
import importlib

from iteras import registry


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


def import_class(classname, additional_modules=None):
    '''
    Dynamically import a class by name from the registered packages.

    Given a class name in CamelCase, this function attempts to import the class
    from one of the following packages (in order):
        - iteras.processing_objects
        - iteras.data_objects
        - iteras.display
        - the packages registered by plugins (see iteras.registry)
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
    if additional_modules is None:
        additional_modules = []

    modulename = camelcase_to_snakecase(classname)
    module_paths = registry.search_packages() + additional_modules

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
