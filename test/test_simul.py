from __future__ import annotations

import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest
from unittest.mock import patch

import yaml
import copy
from pathlib import PureWindowsPath
from typing import Dict, List

import numpy as np
from specula.simul import Simul, computeTag
from specula.loop_control import LoopControl
from specula.connections import InputValue, InputList
from specula.data_objects.recmat import Recmat
from specula.base_data_obj import BaseDataObj
from specula.lib.utils import import_class as real_import_class
from specula.processing_objects.modalrec_multirate import ModalrecMultirate
from specula.data_objects.iir_filter_data import IirFilterData
from specula.scalar_values import IntValue
from specula.base_processing_obj import InputDesc

class DummyObj:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}

class DummyOutput:
    target_device_idx = -1

class DummyOutputDerived(DummyOutput):
    pass

class DummySimulParams:
    def __init__(self, root_dir='dummy', **_kwargs):
        self.root_dir = root_dir
    def init_logging(self, *args):
        pass


class TestSimul(unittest.TestCase):

    @staticmethod
    def _path_suffix_parts(path):
        return PureWindowsPath(path).parts[-2:]

    def setUp(self):
        self.dummySimul = Simul('dummy.yml')
        self.dummySimul.objs = {
            'a': DummyObj(),
            'b': DummyObj()
        }

    def test_none_object_in_parameter_dict_is_none(self):
        '''
        Test that an "_object" directive in the YAML file
        with a "null" value results in a None value.
        
        We use one of our simplest objects setting
        a harmless parameter to " _object: null"
        '''
        yml = '''
        main:
          class: 'SimulParams'
          root_dir: dummy
          
        test:
          class: 'Source'
          polar_coordinates: [1, 2]
          magnitude: null
          wavelengthInNm: null
        '''
        simul = Simul('dummy.yaml')
        params = yaml.safe_load(yml)
        simul.build_objects(params)

        assert simul.objs['test'].magnitude is None
        assert simul.objs['test'].wavelengthInNm is None

    def test_scalar_input_reference(self):
        '''Test that an input is correctly connected'''
        simul = self.dummySimul
        simul.objs['a'].outputs['out'] = DummyOutputDerived()
        simul.objs['b'].inputs['in'] = InputValue(type=DummyOutput)

        simul.connect_objects({
            'b': {
                'inputs': {
                    'in': 'a.out'
                }
            }
        })

        assert isinstance(simul.objs['b'].inputs['in'].get(-1), DummyOutputDerived)
        
    def test_list_input_reference(self):
        '''Test that a list of inputs is correctly connected'''
        simul = self.dummySimul
        simul.objs['a'].outputs['out1'] = DummyOutputDerived()
        simul.objs['a'].outputs['out2'] = DummyOutputDerived()
        simul.objs['b'].inputs['in'] = InputList(type=DummyOutput)

        simul.connect_objects({
            'b': {
                'inputs': {
                    'in': ['a.out1', 'a.out2']
                }
            }
        })

        val = simul.objs['b'].inputs['in'].get(-1)
        assert isinstance(val, list)
        assert all(isinstance(x, DummyOutputDerived) for x in val)

    def test_missing_output_raises(self):
        simul = self.dummySimul
        simul.objs['a'].outputs = {}

        with self.assertRaises(ValueError):
            simul.connect_objects({
                'a': {'outputs': ['missing']}
            })

    def test_invalid_input_type(self):
        simul = self.dummySimul
        simul.objs['a'].outputs['out'] = DummyOutputDerived()
        simul.objs['b'].inputs['in'] = InputValue(type=DummyOutput)

        with self.assertRaises(ValueError):
            simul.connect_objects({
                'b': {
                    'inputs': {
                        'in': 42
                    }
                }
            })

    def test_type_mismatch(self):
        class WrongType:
            pass

        simul = self.dummySimul
        simul.objs['a'].outputs['out'] = WrongType()
        simul.objs['b'].inputs['in'] = InputValue(type=DummyOutput)

        with self.assertRaises(ValueError):
            simul.connect_objects({
                'b': {'inputs': {'in': 'a.out'}}
            })


    def test_delayed_input(self):
        '''This test checks that the has_delayed_input method of
        Simul returns True if any object has a delayed input with
        the -1 syntax.
        '''
        pars = {
            'obj1': {
                'class': 'WaveGenerator',
                'outputs': ['output']
            },
            'obj2': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in2': 'obj1.output:-1'
                }
            },
            'obj3': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in2': 'obj1.output'
                }
            }
        }

        simul = Simul('dummy.yaml')
        assert simul.has_delayed_output('obj1', pars) == True
        assert simul.has_delayed_output('obj2', pars) == False

    def test_delayed_input_detects_circular_loop(self):

        pars = {
            'obj1': {
                'class': 'WaveGenerator',
                'outputs': ['output']
            },
            'obj2': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in2': 'obj1.output:-1'
                }
            },
            'obj3': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in2': 'obj1.output'
                }
            }      
        }
        simul = Simul('dummy.yaml')

        # Does not raise
        _ = simul.build_trigger_order(pars)

        # These outputs depend on each other
        pars = {
            'obj1': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in1': 'obj2.output:-1'
                },
                'outputs': ['output']
            },
            'obj2': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in2': 'obj1.output:-1'
                }
            },
        }
        # Raises ValueError
        with self.assertRaises(ValueError):
            _ = simul.build_trigger_order(pars)


    def test_combine_params(self):

        original_params = {
            'dm': { 'foo' : 'bar'},
            'dm2': { 'foo2': 'bar2'},
        }
        additional_params1 = {'dm_override_2': { 'foo': 'bar3' } }
        additional_params2 = {'remove_3': ['dm2'] }

        simul = Simul('dummy.yaml')

        # Nothing happens for simul_idx=1 (not referenced in additional_params)
        simul.simul_idx = 1
        params = copy.deepcopy(original_params)
        simul.combine_params(params, additional_params1)
        assert params == original_params

        # DM is overridden
        simul.simul_idx = 2
        params = copy.deepcopy(original_params)
        simul.combine_params(params, additional_params1)
        assert params['dm']['foo'] == 'bar3'              # Changed
        assert params['dm2'] == original_params['dm2']    # Unchanged

        # DM2 is removed
        simul.simul_idx = 3
        params = copy.deepcopy(original_params)
        simul.combine_params(params, additional_params2)
        assert params['dm'] == original_params['dm']      # Unchanged
        assert 'dm2' not in params

    def test_unknown_parameter_raises_value_error(self):
        '''Test that a YAML parameter not present in the class constructor raises ValueError'''
        yml = '''
        main:
          class: 'SimulParams'
          root_dir: dummy

        test:
          class: 'Source'
          polar_coordinates: [1, 2]
          magnitude: 1.0
          wavelengthInNm: 500.0
          nonexistent_param: value
        '''
        simul = Simul('dummy.yaml')
        params = yaml.safe_load(yml)
        with self.assertRaises(ValueError):
            simul.build_objects(params)

    def test_data_suffix_stripped_and_null_passed(self):
        '''
        Test that a _data suffix in a YAML key is stripped to match the constructor
        argument name, and that a null value passes None to that argument.
        E.g. slopes_data: null strips to slopes, passing None to Slopes(slopes=None).
        '''
        yml = '''
        main:
          class: 'SimulParams'
          root_dir: dummy

        test:
          class: 'Slopes'
          length: 10
          slopes_data: null
        '''
        simul = Simul('dummy.yaml')
        params = yaml.safe_load(yml)
        simul.build_objects(params)
        # slopes_data: null → strips _data suffix → slopes=None → Slopes initializes as zeros
        assert simul.objs['test'].slopes.shape == (10,)

    def test_overrides(self):
        yml = '''
        main:
          class: 'SimulParams'
          root_dir: dummy
          time_step: 0.001
          total_time: 0.1  
        
        test:
          class: 'Slopes'
          length: 10
          slopes_data: null
          inputs:
            in_pixels: 1.0
        '''
        simul = Simul('dummy.yaml')
        params = yaml.safe_load(yml)
        simul.build_objects(params)

        simul.overrides = ("{main.total_time: 0.2, test.inputs.in_pixels: 2.0}")
        simul.apply_overrides(params)
        simul.build_objects(params)
        assert simul.objs['main'].total_time == 0.2

        simul.overrides = ("{test.inputs.in.pixels: 2.0}")
        with self.assertRaises(ValueError):
            simul.apply_overrides(params)

    def test_ref_suffix_resolves_referenced_object(self):
        '''
        Test that a _ref suffix in a YAML key is stripped and the value is resolved
        by looking up the named object already present in simul.objs.
        Uses IirFilter with simul_params_ref and iir_filter_data_ref to exercise:
        - simul_params_ref → strips to simul_params (plain arg)
        - iir_filter_data_ref → strips to iir_filter_data (arg that itself ends in _data)
        '''
        yml = '''
        main:
          class: 'SimulParams'
          root_dir: dummy
          time_step: 0.001
          total_time: 0.1

        iir_data:
          class: 'IirFilterData'
          ordnum: [2]
          ordden: [2]
          num: [[0.0, 0.3]]
          den: [[-1.0, 1.0]]

        control:
          class: 'IirFilter'
          iir_filter_data_ref: iir_data
          delay: 0
        '''
        simul = Simul('dummy.yaml')
        params = yaml.safe_load(yml)
        simul.build_objects(params)

        assert isinstance(simul.objs['control'].iir_filter_data, IirFilterData)
        assert simul.objs['control'].iir_filter_data is simul.objs['iir_data']

    def test_direct_constructor_arg_ending_in_data(self):
        '''
        Test that a constructor arg that itself ends in _data (e.g. foo_data) is passed
        directly via the else branch, NOT routed to FITS-file reading.
        The parname != name guard on the _data branch ensures this: when no suffix was
        stripped (parname == name), the value is assigned directly.
        '''

        class ClassWithDirectDataArg(BaseDataObj):
            def __init__(self, foo_data=None, target_device_idx=None, precision=None):
                super().__init__(target_device_idx=target_device_idx, precision=precision)
                self.foo_data = foo_data

        def mock_import(classname, additional_modules=None):
            if classname == 'ClassWithDirectDataArg':
                return ClassWithDirectDataArg
            return real_import_class(classname, additional_modules)

        yml = '''
        main:
          class: 'SimulParams'
          root_dir: dummy

        test:
          class: 'ClassWithDirectDataArg'
          foo_data: direct_value
        '''
        with patch('specula.simul.import_class', side_effect=mock_import):
            simul = Simul('dummy.yaml')
            params = yaml.safe_load(yml)
            simul.build_objects(params)
            # foo_data is a direct constructor arg; it should be passed directly,
            # not routed to FITS-file reading (which would fail or mangled the value)
            assert simul.objs['test'].foo_data == 'direct_value'

    def test_dict_object_suffix_stripped_and_loaded(self):
        '''
        Test generic _dict_object behavior:
        - recmat_dict_object strips to constructor arg recmat_dict
        - each dict value is treated as a tag and restored as the hinted object type
        '''
        class ClassWithDictObjectArg(BaseDataObj):
            def __init__(self,
                         recmat_dict: Dict[str, Recmat],
                         target_device_idx=None,
                         precision=None):
                super().__init__(target_device_idx=target_device_idx, precision=precision)
                self.recmat_dict = recmat_dict

        def mock_import(classname, additional_modules=None):
            if classname == 'SimulParams':
                return DummySimulParams
            if classname == 'ClassWithDictObjectArg':
                return ClassWithDictObjectArg
            return real_import_class(classname, additional_modules)

        rec_a = Recmat(np.ones((2, 2), dtype=np.float32), target_device_idx=-1, precision=0)
        rec_b = Recmat(np.full((2, 2), 2.0, dtype=np.float32), target_device_idx=-1, precision=0)

        params = {
            'main': {
                'class': 'SimulParams',
                'root_dir': 'dummy'
            },
            'test': {
                'class': 'ClassWithDictObjectArg',
                'target_device_idx': -1,
                'precision': 0,
                'recmat_dict_object': {
                    'rec_v10': 'tag_fast',
                    'rec_v01': 'tag_slow'
                }
            }
        }

        with patch('specula.simul.import_class', side_effect=mock_import):
            with patch('specula.data_objects.recmat.Recmat.restore', side_effect=[rec_a, rec_b]) as mock_restore:
                simul = Simul('dummy.yaml')
                simul.build_objects(params)

                obj = simul.objs['test']
                assert set(obj.recmat_dict.keys()) == {'rec_v10', 'rec_v01'}
                assert isinstance(obj.recmat_dict['rec_v10'], Recmat)
                assert isinstance(obj.recmat_dict['rec_v01'], Recmat)

                assert mock_restore.call_count == 2
                first_path = mock_restore.call_args_list[0].args[0]
                second_path = mock_restore.call_args_list[1].args[0]
                assert self._path_suffix_parts(first_path) == ('rec', 'tag_fast.fits')
                assert self._path_suffix_parts(second_path) == ('rec', 'tag_slow.fits')

    def test_dict_object_raises_with_no_type(self):

        class ClassWithDictObjectArg(BaseDataObj):
            def __init__(self,
                         recmat_dict: Recmat=None,      # Wrong type, should be Dict[str, Recmat]
                         target_device_idx=None,
                         precision=None):
                # Will not be instantiated
                pass

        def mock_import(classname, additional_modules=None):
            if classname == 'SimulParams':
                return DummySimulParams
            if classname == 'ClassWithDictObjectArg':
                return ClassWithDictObjectArg

        rec_a = Recmat(np.ones((2, 2), dtype=np.float32), target_device_idx=-1, precision=0)
        rec_b = Recmat(np.full((2, 2), 2.0, dtype=np.float32), target_device_idx=-1, precision=0)

        params = {
            'main': {
                'class': 'SimulParams',
                'root_dir': 'dummy'
            },
            'test': {
                'class': 'ClassWithDictObjectArg',
                'target_device_idx': -1,
                'precision': 0,
                'recmat_dict_object': {
                    'rec_v10': 'tag_fast',
                    'rec_v01': 'tag_slow'
                }
            }
        }

        with patch('specula.simul.import_class', side_effect=mock_import):
            with patch('specula.data_objects.recmat.Recmat.restore', side_effect=[rec_a, rec_b]) as mock_restore:
                simul = Simul('foo.yml')
                with self.assertRaises(ValueError):
                    simul.build_objects(params)

    def test_list_object_suffix_stripped_and_loaded(self):
        '''
        Test generic _list_object behavior:
        - recmat_list_object strips to constructor arg recmat_list
        - each list value is treated as a tag and restored as the hinted object type
        '''
        class ClassWithListObjectArg(BaseDataObj):
            def __init__(self,
                         recmat_list: List[Recmat],
                         target_device_idx=None,
                         precision=None):
                super().__init__(target_device_idx=target_device_idx, precision=precision)
                self.recmat_list = recmat_list

        def mock_import(classname, additional_modules=None):
            if classname == 'SimulParams':
                return DummySimulParams
            if classname == 'ClassWithListObjectArg':
                return ClassWithListObjectArg
            return real_import_class(classname, additional_modules)

        rec_a = Recmat(np.ones((2, 2), dtype=np.float32), target_device_idx=-1, precision=0)
        rec_b = Recmat(np.full((2, 2), 2.0, dtype=np.float32), target_device_idx=-1, precision=0)

        params = {
            'main': {
                'class': 'SimulParams',
                'root_dir': 'dummy'
            },
            'test': {
                'class': 'ClassWithListObjectArg',
                'target_device_idx': -1,
                'precision': 0,
                'recmat_list_object': ['tag_fast', 'tag_slow']
            }
        }

        with patch('specula.simul.import_class', side_effect=mock_import):
            with patch('specula.data_objects.recmat.Recmat.restore', side_effect=[rec_a, rec_b]) as mock_restore:
                simul = Simul('dummy.yaml')
                simul.build_objects(params)

                obj = simul.objs['test']
                assert len(obj.recmat_list) == 2
                assert isinstance(obj.recmat_list[0], Recmat)
                assert isinstance(obj.recmat_list[1], Recmat)
                assert obj.recmat_list[0].tag == 'tag_fast'
                assert obj.recmat_list[1].tag == 'tag_slow'

                assert mock_restore.call_count == 2
                first_path = mock_restore.call_args_list[0].args[0]
                second_path = mock_restore.call_args_list[1].args[0]
                assert self._path_suffix_parts(first_path) == ('rec', 'tag_fast.fits')
                assert self._path_suffix_parts(second_path) == ('rec', 'tag_slow.fits')

    def test_list_object_raises_with_no_type(self):
        '''
        Test generic _list_object behavior:
        - recmat_list_object strips to constructor arg recmat_list
        - each list value is treated as a tag and restored as the hinted object type
        '''

        class ClassWithListObjectArg(BaseDataObj):
            def __init__(self,
                         recmat_list: Recmat=None,     # Wrong type, should be List[Recmat]
                         target_device_idx=None,
                         precision=None):
                # Will not be instantiated
                pass

        def mock_import(classname, additional_modules=None):
            if classname == 'SimulParams':
                return DummySimulParams
            if classname == 'ClassWithListObjectArg':
                return ClassWithListObjectArg

        rec_a = Recmat(np.ones((2, 2), dtype=np.float32), target_device_idx=-1, precision=0)
        rec_b = Recmat(np.full((2, 2), 2.0, dtype=np.float32), target_device_idx=-1, precision=0)

        params = {
            'main': {
                'class': 'SimulParams',
                'root_dir': 'dummy'
            },
            'test': {
                'class': 'ClassWithListObjectArg',
                'target_device_idx': -1,
                'precision': 0,
                'recmat_list_object': ['tag_fast', 'tag_slow']
            }
        }

        with patch('specula.simul.import_class', side_effect=mock_import):
            with patch('specula.data_objects.recmat.Recmat.restore', side_effect=[rec_a, rec_b]) as mock_restore:
                simul = Simul('foo.yml')
                with self.assertRaises(ValueError):
                    simul.build_objects(params)

    def test_build_targeted_replay_follows_list_ref_dependencies(self):
        params = {
            'main': {'class': 'SimulParams', 'root_dir': 'dummy'},
            'src_a': {'class': 'Source'},
            'src_b': {'class': 'Source'},
            'consumer': {
                'class': 'DummyClass',
                'source_list_ref': ['src_a', 'src_b']
            }
        }

        replay = Simul('dummy.yaml').build_targeted_replay(params, 'consumer')

        assert 'consumer' in replay
        assert 'src_a' in replay
        assert 'src_b' in replay

    def test_integration_simul_modalrec_with_list_object(self):
        '''
        Integration-style test: Simul builds ModalrecMultirate and injects
        recmat_list via _list_object using mocked Recmat.restore.
        '''
        def mock_import(classname, additional_modules=None):
            if classname == 'SimulParams':
                return DummySimulParams
            return real_import_class(classname, additional_modules)

        rec_both = Recmat(np.ones((5, 4), dtype=np.float32), target_device_idx=-1, precision=0)
        rec_s1 = Recmat(np.ones((5, 4), dtype=np.float32), target_device_idx=-1, precision=0)
        rec_s2 = Recmat(np.ones((5, 4), dtype=np.float32), target_device_idx=-1, precision=0)

        params = {
            'main': {
                'class': 'SimulParams',
                'root_dir': 'dummy'
            },
            'rec': {
                'class': 'ModalrecMultirate',
                'target_device_idx': -1,
                'precision': 0,
                'recmat_list_object': ['tag_both', 'tag_s1', 'tag_s2'],
                'validity_masks': [[True, True], [True, False], [False, True]],
                'n_modes_total': 5
            }
        }

        with patch('specula.simul.import_class', side_effect=mock_import):
            with patch('specula.data_objects.recmat.Recmat.restore', side_effect=[rec_both, rec_s1, rec_s2]):
                simul = Simul('dummy.yaml')
                simul.build_objects(params)

                rec_obj = simul.objs['rec']
                assert isinstance(rec_obj, ModalrecMultirate)
                assert set(rec_obj.recmat_by_mask.keys()) == {(True, True), (True, False), (False, True)}


    def test_simul_with_no_yaml_files(self):

        with self.assertRaises(ValueError):
            _ = Simul()

    def test_exception_raised_when_extra_parameter_with_tag(self):
        yml = '''
        main:
          class: 'SimulParams'
          root_dir: dummy
          
        test:
          class: 'Pupilstop'
          tag: 'abcdef'
          foo: 42
        '''
        simul = Simul('dummy.yaml')
        params = yaml.safe_load(yml)

        with self.assertRaises(ValueError):
            simul.build_objects(params)

    def test_exception_raised_when_restoring_with_no_type_hint(self):
        yml = '''
        main:
          class: 'SimulParams'
          root_dir: dummy
          
        test:
          class: 'Slopes'
          slopes_object: 'foo'
        '''
        simul = Simul('dummy.yaml')
        params = yaml.safe_load(yml)

        with self.assertRaises(ValueError):
            simul.build_objects(params)

    def test_compute_tag(self):
        '''Test that even small changes in names result in a different tag'''

        output_obj_name = 'foo'
        dest_object = 'bar'
        output_attr_name = 'test1'
        input_attr_name = 'test2'

        tag1 = computeTag(output_obj_name, dest_object, output_attr_name, input_attr_name)

        output_obj_name = 'foo2'
        tag2 = computeTag(output_obj_name, dest_object, output_attr_name, input_attr_name)

        dest_object = 'bar2'
        tag3 = computeTag(output_obj_name, dest_object, output_attr_name, input_attr_name)

        output_attr_name = 'atest1'
        tag4 = computeTag(output_obj_name, dest_object, output_attr_name, input_attr_name)

        input_attr_name = 'atest2'
        tag5 = computeTag(output_obj_name, dest_object, output_attr_name, input_attr_name)

        assert len(set((tag1, tag2, tag3, tag4, tag5))) == 5

    def test_target_device_idx(self):

        yml = '''
        main:
          class: 'SimulParams'
          root_dir: dummy

        test:
          class: 'Slopes'
          length: 10
          slopes_data: null
          target_device_idx: -1
        '''
        simul = Simul('dummy.yaml')
        params = yaml.safe_load(yml)
        simul.build_objects(params)
        assert simul.objs['test'].target_device_idx == -1


class TestSimulRunTiming(unittest.TestCase):
    """Tests that Simul.run() translates start_time/end_time correctly to loop.run()"""

    _MINIMAL_YML = '''\
main:
  class: SimulParams
  root_dir: dummy
  total_time: 1.0
  time_step: 0.001
'''

    def _make_simul(self):
        fd, path = tempfile.mkstemp(suffix='.yml')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(self._MINIMAL_YML)
            return Simul(path, speed_report=False), path
        except Exception:
            os.unlink(path)
            raise

    def _run_and_capture(self, simul, **run_kwargs):
        captured = {}
        original_run = LoopControl.run

        def fake_run(self_loop, run_time, dt, t0=0, speed_report=False):
            captured['run_time'] = run_time
            captured['t0'] = t0

        with patch.object(LoopControl, 'run', fake_run):
            simul.run(**run_kwargs)
        return captured

    def test_default_call_passes_total_time_and_zero_t0(self):
        """Default run() → run_time=total_time, t0=0"""
        simul, path = self._make_simul()
        try:
            captured = self._run_and_capture(simul)
        finally:
            os.unlink(path)
        self.assertAlmostEqual(captured['run_time'], 1.0)
        self.assertAlmostEqual(captured['t0'], 0.0)

    def test_start_time_reduces_run_time_and_sets_t0(self):
        """start_time=0.1, no end_time → run_time=0.9, t0=0.1"""
        simul, path = self._make_simul()
        try:
            captured = self._run_and_capture(simul, start_time=0.1)
        finally:
            os.unlink(path)
        self.assertAlmostEqual(captured['run_time'], 0.9)
        self.assertAlmostEqual(captured['t0'], 0.1)

    def test_end_time_limits_run_time(self):
        """start_time=0.1, end_time=0.8 → run_time=0.7, t0=0.1"""
        simul, path = self._make_simul()
        try:
            captured = self._run_and_capture(simul, start_time=0.1, end_time=0.8)
        finally:
            os.unlink(path)
        self.assertAlmostEqual(captured['run_time'], 0.7)
        self.assertAlmostEqual(captured['t0'], 0.1)
