'''
Registry of the packages that extend iteras (plugins).

A plugin is a package providing data objects, processing objects and
displays for a specific domain. It registers itself calling
:func:`register_plugin`, usually when its top-level package is imported.
The iteras command line also loads the plugins declared with an
``iteras.plugins`` entry point (see :func:`load_entry_point_plugins`).

The registry holds:

- the packages searched, in order, when a class is looked up by name
  (see :func:`iteras.lib.utils.import_class`);
- the top-level package names of the plugins, used as logger roots;
- the calibration subdirectories of the data types (see :class:`iteras.calib_manager.CalibManager`);
- alternative names of iteras's environment variables (see :func:`getenv`);
- the application name shown to the user (prompts, messages), see :func:`set_application_name`.

This module must stay lightweight: it is imported by iteras's ``__init__``.
'''

# Packages searched by import_class(), in order
_packages = ['iteras.processing_objects',
             'iteras.data_objects',
             'iteras.display']

# Top-level packages of iteras and its plugins
_plugin_roots = ['iteras']

# Calibration subdirectories, by data type or class name
_calib_subdirs = {
    'data': 'data/',
}

# Alternative names of iteras's environment variables, {iteras name: [alias, ...]}
_env_aliases = {}

_application_name = 'iteras'

_entry_points_loaded = False


def register_plugin(root, packages=(), calib_subdirs=None, env_aliases=None):
    '''
    Register a plugin.

    Parameters
    ----------
    root : str
        Top-level package name of the plugin, used as logger root
        for its classes.
    packages : list of str
        Packages searched by import_class(), after the ones already registered.
    calib_subdirs : dict, optional
        Calibration subdirectories, see register_calib_subdirs().
    env_aliases : dict, optional
        Alternative names of iteras's environment variables, see register_env_alias().
    '''
    if root not in _plugin_roots:
        _plugin_roots.append(root)
    register_packages(*packages)
    if calib_subdirs:
        register_calib_subdirs(calib_subdirs)
    for name, alias in (env_aliases or {}).items():
        register_env_alias(name, alias)


def register_packages(*packages):
    '''Add packages to the ones searched by import_class(), keeping the registration order'''
    for package in packages:
        if package not in _packages:
            _packages.append(package)


def register_calib_subdirs(subdirs):
    '''
    Add calibration subdirectories. *subdirs* maps a data type or class
    name (as used by CalibManager) to a subdirectory of the calibration root.
    '''
    _calib_subdirs.update(subdirs)


def register_env_alias(name, alias):
    '''
    Register *alias* as an alternative name of the iteras environment variable *name*
    (for example MYAPP_DISABLE_GPU for ITERAS_DISABLE_GPU), see getenv().
    '''
    aliases = _env_aliases.setdefault(name, [])
    if alias not in aliases:
        aliases.append(alias)


def getenv(name, default=None):
    '''
    Value of the iteras environment variable *name*, or of the first
    of its aliases that is set, or *default*. Read at each call.
    '''
    import os
    for key in [name] + _env_aliases.get(name, []):
        if key in os.environ:
            return os.environ[key]
    return default


def set_application_name(name):
    '''Set the application name shown to the user, for example in prompts and messages'''
    global _application_name
    _application_name = name


def application_name():
    '''Application name shown to the user: 'iteras', unless a plugin changed it'''
    return _application_name


def search_packages():
    '''Packages searched by import_class(), in order'''
    return list(_packages)


def plugin_roots():
    '''Top-level packages of iteras and of the registered plugins'''
    return list(_plugin_roots)


def calib_subdirs():
    '''Calibration subdirectories of all registered data types'''
    return dict(_calib_subdirs)


def load_entry_point_plugins():
    '''
    Import the plugins declared with an ``iteras.plugins`` entry point,
    so that they register themselves. Called once, by the iteras command line.
    '''
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    from importlib.metadata import entry_points
    for ep in entry_points(group='iteras.plugins'):
        ep.load()
