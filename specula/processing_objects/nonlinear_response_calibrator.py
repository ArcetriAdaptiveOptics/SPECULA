import os

import numpy as np

from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.slopes import Slopes
from specula.data_objects.recmat import Recmat
from specula.data_objects.nonlinear_calibration import NonlinearCalibration


class NonlinearResponseCalibrator(BaseProcessingObj):
    """
    Measures the per-mode nonlinear response of a wavefront sensor and saves
    it as a :class:`~specula.data_objects.nonlinear_calibration.NonlinearCalibration`,
    for :class:`~specula.processing_objects.nonlinear_modalrec.NonlinearModalrec`.

    The DM must be driven one mode at a time over a grid of amplitudes, for
    example by a :class:`~specula.processing_objects.push_pull_generator.PushPullGenerator`
    with ``push_pull_type: 'PUSHPULL'``, ``constant_amp: true`` and the
    amplitude grid as ``pattern``. At each step the measured slopes are
    reconstructed with the row of `recmat` of the active mode, which gives that
    mode's "response" at the commanded amplitude: ``(recmat @ slopes)[mode]``.
    It is the linear estimate of NonlinearModalrec, which must use the same
    recmat, so that inverting the curve recovers the amplitude of a mode
    injected alone, even when the saturated slopes change shape. At
    :meth:`finalize`, the responses are averaged over repeated amplitudes and
    saved.

    The curves depend on the residual wavefront during the calibration, as
    the optical gains do: a calibration without turbulence overestimates the
    response in closed loop.

    As in :class:`~specula.processing_objects.im_calibrator.ImCalibrator`, the
    active mode and its amplitude are the non-zero entry of `in_commands`, so
    the amplitude grid must not include zero.

    Parameters
    ----------
    nmodes : int
        Number of calibrated modes: first_mode, ..., first_mode + nmodes - 1.
    recmat : Recmat
        Reconstruction matrix of NonlinearModalrec; its rows are indexed as the
        commands (row k reconstructs command k).
    data_dir : str
        Directory of the calibration file (set by the calibration manager).
    calib_tag : str
        Name of the calibration file.
    first_mode : int
        Index, in the command vector, of the first calibrated mode.
    overwrite : bool
        Overwrite an existing calibration file.
    """

    def __init__(self,
                 nmodes: int,
                 recmat: Recmat,
                 data_dir: str,
                 calib_tag: str,
                 first_mode: int = 0,
                 overwrite: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if first_mode + nmodes > recmat.nmodes:
            raise ValueError(f'modes {first_mode}..{first_mode + nmodes - 1} are not all in the recmat, '
                             f'which has {recmat.nmodes} modes')
        self.nmodes = nmodes
        self.first_mode = first_mode
        self.data_dir = data_dir
        self.overwrite = overwrite

        self.calib_path = os.path.join(data_dir, calib_tag)
        if not self.calib_path.endswith('.fits'):
            self.calib_path += '.fits'
        if os.path.exists(self.calib_path) and not overwrite:
            raise FileExistsError(f'Calibration file {self.calib_path} already exists, please remove it')

        # Recmat rows of the calibrated modes: the response is the dot product with the slopes
        self._projectors = self.to_xp(recmat.recmat[first_mode:first_mode + nmodes], dtype=self.dtype)
        if self.xp.any(self.xp.all(self._projectors == 0, axis=1)):
            raise ValueError('the recmat has all-zero rows among the calibrated modes')

        self._amplitude_samples = [[] for _ in range(nmodes)]
        self._response_samples = [[] for _ in range(nmodes)]

        self.inputs['in_slopes'] = InputValue(type=Slopes)
        self.inputs['in_commands'] = InputValue(type=BaseValue)

    @classmethod
    def input_names(cls):
        return {'in_slopes': InputDesc(Slopes, 'Measured slopes during the amplitude sweep'),
                'in_commands': InputDesc(BaseValue, 'Commanded modal amplitude vector')}

    @classmethod
    def output_names(cls):
        return {}

    def trigger_code(self):
        commands = self.local_inputs['in_commands'].value
        idx = self.xp.nonzero(commands)[0]
        if len(idx) == 0:
            return

        mode = int(idx[0]) - self.first_mode
        if mode < 0 or mode >= self.nmodes:
            return

        slopes = self.to_xp(self.local_inputs['in_slopes'].slopes)
        self._amplitude_samples[mode].append(float(commands[idx[0]]))
        self._response_samples[mode].append(float(self.xp.dot(slopes, self._projectors[mode])))

    def finalize(self):
        # The amplitude grid: all the amplitudes sampled, the same for every mode
        shared_amplitudes = sorted(set(
            round(a, 12) for samples in self._amplitude_samples for a in samples
        ))
        if not shared_amplitudes:
            raise RuntimeError('No calibration samples were collected: '
                               'check that the driving generator actually excited every mode.')
        shared_amplitudes = np.array(shared_amplitudes)

        responses = np.zeros((self.nmodes, len(shared_amplitudes)))
        for mode in range(self.nmodes):
            amps = np.round(np.array(self._amplitude_samples[mode]), 12)
            resp = np.array(self._response_samples[mode])
            for j, a in enumerate(shared_amplitudes):
                sel = amps == a
                if not np.any(sel):
                    raise RuntimeError(f'Mode {mode + self.first_mode} was not sampled at amplitude {a}; '
                                       'every mode must be swept over the same amplitude grid.')
                responses[mode, j] = resp[sel].mean()

        calib = NonlinearCalibration(shared_amplitudes, responses, first_mode=self.first_mode,
                                     target_device_idx=self.target_device_idx)
        bad = calib.non_monotonic_modes()
        if bad:
            self.logger.warning(f'the response of modes {bad} is not strictly increasing with the '
                                f'amplitude: NonlinearModalrec cannot use this calibration. Reduce the '
                                f'largest amplitudes, or calibrate fewer modes')
        os.makedirs(self.data_dir, exist_ok=True)
        calib.save(self.calib_path, overwrite=self.overwrite)
        super().finalize()
