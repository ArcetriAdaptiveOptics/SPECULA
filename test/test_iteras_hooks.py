'''
Methods that plugins can override to change how the framework treats
their objects (see the "Extension API" documentation page).
'''
import specula
specula.init(0)  # Default target device

import os
import sys
import tempfile
import textwrap
import unittest

import yaml

from iteras.simul import Simul
from iteras.base_data_obj import BaseDataObj
from iteras.base_processing_obj import BaseProcessingObj
from iteras.base_value import BaseValue
from iteras.display.base_display import BaseDisplay
from iteras.display.plot_display import PlotDisplay
from iteras.display.plot_vector_display import PlotVectorDisplay
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.layer import Layer


class TestClassHooks(unittest.TestCase):

    def test_defaults(self):
        self.assertFalse(BaseProcessingObj.accepts_input_list())
        self.assertFalse(BaseDataObj.track_consumers_in_replay())
        self.assertFalse(BaseValue.track_consumers_in_replay())

    def test_overrides(self):
        self.assertTrue(ElectricField.track_consumers_in_replay())
        self.assertTrue(Layer.track_consumers_in_replay())
        # The methods do not use the instance
        self.assertTrue(BaseDisplay.can_skip_updates(None))
        self.assertFalse(PlotDisplay.can_skip_updates(None))
        self.assertFalse(PlotVectorDisplay.can_skip_updates(None))


class TestSimulHooks(unittest.TestCase):
    '''
    Run a short simulation with objects from a temporary package, and check
    that Simul calls their hooks.
    '''

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        pkg = os.path.join(root, 'hooks_objs')
        os.makedirs(pkg)
        open(os.path.join(pkg, '__init__.py'), 'w').close()
        with open(os.path.join(pkg, 'counter.py'), 'w') as f:
            f.write(textwrap.dedent('''
                from iteras.base_processing_obj import BaseProcessingObj
                from iteras.base_value import BaseValue

                class Counter(BaseProcessingObj):
                    def __init__(self, target_device_idx=None, precision=None):
                        super().__init__(target_device_idx=target_device_idx, precision=precision)
                        self.out = BaseValue(value=self.xp.zeros(1, dtype=self.dtype),
                                             target_device_idx=target_device_idx)
                        self.outputs['out_value'] = self.out

                    def trigger_code(self):
                        self.out.value += 1

                    def post_trigger(self):
                        super().post_trigger()
                        self.out.generation_time = self.current_time
                '''))
        with open(os.path.join(pkg, 'recorder.py'), 'w') as f:
            f.write(textwrap.dedent('''
                from iteras.base_processing_obj import BaseProcessingObj

                calls = []

                class Recorder(BaseProcessingObj):
                    """Not a DataStore, but accepts an input_list like it"""
                    @classmethod
                    def accepts_input_list(cls):
                        return True

                    def input_list_created(self):
                        calls.append(('input_list_created', sorted(self.inputs)))

                    def set_simulation_params(self, params):
                        calls.append(('set_simulation_params', sorted(params)))

                    def set_replay_params(self, replay_params):
                        calls.append(('set_replay_params', sorted(replay_params)))

                    def check_input_names(self):
                        pass
                '''))
        with open(os.path.join(pkg, 'my_store.py'), 'w') as f:
            f.write(textwrap.dedent('''
                from iteras.processing_objects.data_store import DataStore

                class MyStore(DataStore):
                    """A subclass of DataStore is treated as a DataStore"""
                '''))
        sys.path.insert(0, root)
        self.addCleanup(sys.path.remove, root)
        self.addCleanup(lambda: [sys.modules.pop(m) for m in list(sys.modules)
                                 if m.split('.')[0] == 'hooks_objs'])

        self.store_dir = os.path.join(root, 'store')
        self.params = {
            'main': {'class': 'SimParams', 'root_dir': root, 'total_time': 0.003,
                     'time_step': 0.001, 'add_modules': ['hooks_objs']},
            'counter': {'class': 'Counter'},
            'rec': {'class': 'Recorder', 'inputs': {'input_list': ['v-counter.out_value']}},
            'store': {'class': 'MyStore', 'store_dir': self.store_dir, 'create_tn': False,
                      'inputs': {'input_list': ['v-counter.out_value']}},
        }
        self.yml = os.path.join(root, 'params.yml')
        with open(self.yml, 'w') as f:
            yaml.dump(self.params, f)

    def test_hooks_are_called(self):
        simul = Simul(self.yml, speed_report=False)
        simul.run()

        from hooks_objs.recorder import calls
        names = [c[0] for c in calls]
        self.assertEqual(names, ['set_simulation_params', 'input_list_created', 'set_replay_params'])
        self.assertEqual(calls[0][1], sorted(self.params))
        # The input_list entry is replaced by one input per item, named after its alias
        self.assertEqual(calls[1][1], ['v'])
        # The replay replaces the DataStore subclass with a DataSource
        self.assertIn('data_source', calls[2][1])
        self.assertNotIn('store', calls[2][1])

        # The DataStore subclass received its inputs and parameters, and stored the data
        self.assertEqual(sorted(simul.objs['store'].inputs), ['v'])
        self.assertTrue(os.path.exists(os.path.join(self.store_dir, 'v.fits')))
        self.assertTrue(os.path.exists(os.path.join(self.store_dir, 'replay_params.yml')))


if __name__ == '__main__':
    unittest.main()
