import os
import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray, np
from specula.data_objects.source import Source
from specula.processing_objects.atmo_random_phase import AtmoRandomPhase
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.pupilstop import Pupilstop
from specula.base_value import BaseValue

from test.specula_testlib import cpu_and_gpu


class TestAtmoRandomPhase(unittest.TestCase):

    @cpu_and_gpu
    def test_output_ef_have_the_correct_names(self, target_device_idx, xp):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)
        lgs1_source = Source( polar_coordinates=[45.0, 0.0], height=90000, magnitude=5, wavelengthInNm=589)

        simulParams = SimulParams(pixel_pupil=160, pixel_pitch=0.05)
        atmo = AtmoRandomPhase(simulParams,
                            L0=23,  # [m] Outer scale
                            data_dir=data_dir,
                            source_dict = {'on_axis_source': on_axis_source,
                                            'lgs1_source': lgs1_source},
                            target_device_idx=target_device_idx)

        assert len(atmo.outputs) == 2
        assert 'out_on_axis_source_ef' in atmo.outputs
        assert 'out_lgs1_source_ef' in atmo.outputs

    @cpu_and_gpu
    def test_update_interval_default(self, target_device_idx, xp):
        """Test that by default (update_interval=1) the phase changes every step"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)

        simul_params = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=0.001)
        atmo = AtmoRandomPhase(simul_params,
                            L0=23,
                            data_dir=data_dir,
                            source_dict={'on_axis_source': on_axis_source},
                            update_interval=1,  # Default: update every step
                            target_device_idx=target_device_idx)

        # Setup inputs
        pupilstop = Pupilstop(simul_params, target_device_idx=target_device_idx)
        seeing = BaseValue(value=xp.array([0.8]),
                           target_device_idx=target_device_idx)

        atmo.inputs['pupilstop'].set(pupilstop)
        atmo.inputs['seeing'].set(seeing)
        atmo.setup()

        # Run for 3 steps and collect phases
        phases = []
        for step in range(3):
            current_time = atmo.seconds_to_t(step * 0.001)
            seeing.generation_time = current_time
            pupilstop.generation_time = current_time

            atmo.check_ready(current_time)
            atmo.trigger()
            atmo.post_trigger()

            phases.append(cpuArray(atmo.outputs['out_on_axis_source_ef'].phaseInNm.copy()))

        # With update_interval=1, each phase should be different
        self.assertFalse(np.allclose(phases[0], phases[1]))
        self.assertFalse(np.allclose(phases[1], phases[2]))
        self.assertFalse(np.allclose(phases[0], phases[2]))

    @cpu_and_gpu
    def test_update_interval_freeze(self, target_device_idx, xp):
        """Test that with large update_interval the phase stays frozen"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)

        simul_params = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=0.001)
        atmo = AtmoRandomPhase(simul_params,
                            L0=23,
                            data_dir=data_dir,
                            source_dict={'on_axis_source': on_axis_source},
                            update_interval=100,  # Update only every 100 steps
                            target_device_idx=target_device_idx)

        # Setup inputs
        pupilstop = Pupilstop(simul_params, target_device_idx=target_device_idx)
        seeing = BaseValue(value=xp.array([0.8]),
                           target_device_idx=target_device_idx)

        atmo.inputs['pupilstop'].set(pupilstop)
        atmo.inputs['seeing'].set(seeing)
        atmo.setup()

        # Run for 5 steps (less than update_interval)
        phases = []
        for step in range(5):
            current_time = atmo.seconds_to_t(step * 0.001)
            seeing.generation_time = current_time
            pupilstop.generation_time = current_time

            atmo.check_ready(current_time)
            atmo.trigger()
            atmo.post_trigger()

            phases.append(cpuArray(atmo.outputs['out_on_axis_source_ef'].phaseInNm.copy()))

        # All phases should be identical (same screen, only scaling might differ slightly)
        # We compare the normalized phases to avoid scaling differences
        for i in range(1, len(phases)):
            phase_ratio = phases[i] / (phases[0] + 1e-10)  # Avoid division by zero
            # The ratio should be constant (close to 1.0) everywhere
            self.assertTrue(np.allclose(phase_ratio, phase_ratio[0, 0], rtol=1e-6))

    @cpu_and_gpu
    def test_update_interval_specific(self, target_device_idx, xp):
        """Test that phase updates at the correct intervals"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)

        simul_params = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=0.001)
        update_interval = 3
        atmo = AtmoRandomPhase(simul_params,
                            L0=23,
                            data_dir=data_dir,
                            source_dict={'on_axis_source': on_axis_source},
                            update_interval=update_interval,
                            target_device_idx=target_device_idx)

        # Setup inputs
        pupilstop = Pupilstop(simul_params, target_device_idx=target_device_idx)
        seeing = BaseValue(value=xp.array([0.8]),
                           target_device_idx=target_device_idx)

        atmo.inputs['pupilstop'].set(pupilstop)
        atmo.inputs['seeing'].set(seeing)
        atmo.setup()

        # Run for 10 steps and track when position changes
        positions = []
        for step in range(10):
            current_time = atmo.seconds_to_t(step * 0.001)
            seeing.generation_time = current_time
            pupilstop.generation_time = current_time

            atmo.check_ready(current_time)
            atmo.trigger()
            atmo.post_trigger()

            positions.append(atmo.last_position)

        # Position should change every update_interval steps
        # Step 0,1: step_counter=1,2 (1%3!=0, 2%3!=0) -> position 0
        # Step 2: step_counter=3 (3%3==0) -> increment to position 1
        # Step 3,4: step_counter=4,5 (4%3!=0, 5%3!=0) -> position 1
        # Step 5: step_counter=6 (6%3==0) -> increment to position 2
        # Step 6,7: step_counter=7,8 (7%3!=0, 8%3!=0) -> position 2
        # Step 8: step_counter=9 (9%3==0) -> increment to position 3
        # Step 9: step_counter=10 (10%3!=0) -> position 3
        expected_positions = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
        self.assertEqual(positions, expected_positions)

    @cpu_and_gpu
    def test_update_interval_with_seed_reset(self, target_device_idx, xp):
        """Test that when running out of screens, new ones are generated with incremented seed"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        on_axis_source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)

        # Use small pixel_phasescreens to quickly run out of screens
        simul_params = SimulParams(pixel_pupil=160, pixel_pitch=0.05, time_step=0.001)
        initial_seed = 42
        atmo = AtmoRandomPhase(simul_params,
                            L0=23,
                            data_dir=data_dir,
                            source_dict={'on_axis_source': on_axis_source},
                            pixel_phasescreens=320,  # Only 2 screens (320/160 = 2)
                            seed=initial_seed,
                            update_interval=1,
                            target_device_idx=target_device_idx)

        # Setup inputs
        pupilstop = Pupilstop(simul_params, target_device_idx=target_device_idx)
        seeing = BaseValue(value=xp.array([0.8]),
                           target_device_idx=target_device_idx)

        atmo.inputs['pupilstop'].set(pupilstop)
        atmo.inputs['seeing'].set(seeing)
        atmo.setup()

        initial_num_screens = atmo.phasescreens.shape[0]

        # Run for enough steps to exhaust all screens
        for step in range(initial_num_screens + 2):
            current_time = atmo.seconds_to_t(step * 0.001)
            seeing.generation_time = current_time
            pupilstop.generation_time = current_time

            atmo.check_ready(current_time)
            atmo.trigger()
            atmo.post_trigger()

        # Seed should have been incremented
        self.assertEqual(atmo.seed, initial_seed + 1)
        # Position should have reset
        self.assertLess(atmo.last_position, initial_num_screens)
