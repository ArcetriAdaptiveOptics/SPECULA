import pytest
import logging

from specula.simul import Simul
from specula.log import get_level_names_mapping


class DummyProcessingObject:
    def __init__(self, name=None):
        if name is not None:
            self.name = name
        self.last_called = None

    def init_logging(self, name, level):
        # record what init_logging was called with
        self.last_called = (name, level)


def make_simul(log_level=logging.INFO):
    # Simul only requires at least one param file string for construction
    return Simul("params.yml", log_level=log_level)


def test_obj_init_logging_with_verbose_true():
    s = make_simul()
    d = DummyProcessingObject('myobj')
    pars = {'verbose': True}
    s.obj_init_logging(d, pars)
    assert d.last_called == ('myobj', logging.DEBUG)
    assert 'verbose' not in pars


def test_obj_init_logging_without_verbose_false():
    s = make_simul()
    d = DummyProcessingObject()
    pars = {'verbose': False}
    s.obj_init_logging(d, pars)
    assert d.last_called == (None, logging.INFO)
    assert 'verbose' not in pars


def test_obj_init_logging_with_int():
    s = make_simul()
    d = DummyProcessingObject('i')
    pars = {'verbose': 15}
    s.obj_init_logging(d, pars)
    assert d.last_called == ('i', 15)


def test_obj_init_logging_with_str():
    s = make_simul()
    d = DummyProcessingObject('s')
    pars = {'verbose': 'warning'}
    levels = get_level_names_mapping()
    expected = levels['WARNING']
    s.obj_init_logging(d, pars)
    assert d.last_called == ('s', expected)


def test_obj_init_logging_invalid_value():
    s = make_simul()
    d = DummyProcessingObject(name='bad')
    pars = {'verbose': [1, 2, 3]}   # a list is not a valid log level
    with pytest.raises(ValueError):
        s.obj_init_logging(d, pars)


def test_obj_init_logging_default_level():
    # If no verbose is provided, the simul logger effective level is used
    s = make_simul(log_level=logging.DEBUG)
    d = DummyProcessingObject('def')
    pars = {}
    s.obj_init_logging(d, pars)
    assert d.last_called == ('def', logging.DEBUG)
