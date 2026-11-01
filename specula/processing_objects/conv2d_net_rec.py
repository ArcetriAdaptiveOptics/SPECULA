import numpy as np

from specula.base_processing_obj import InputDesc
from specula.processing_objects.base_modalrec import BaseModalrec
from specula.connections import InputValue
from specula.base_value import BaseValue
from iteras.lib.frame_stacker import FrameStacker

# torch is optional (the "nn" extra in pyproject.toml): this module must
# import without it, so that SPECULA itself does; the constructor raises.
try:
    import torch
    from iteras.lib.cnn_checkpoint import calibration_factor, load_trained_network, network_input
    TORCH_IMPORT_ERROR = None
except ImportError as e:
    TORCH_IMPORT_ERROR = e


class Conv2dNetRec(BaseModalrec):
    """
    Closed-loop-usable, deterministic inference reconstructor: loads a
    network trained by :class:`~iteras.processing_objects.conv2d_net_trainer.Conv2dNetTrainer`
    and reconstructs modes from a 2D map built from the slopes, on every
    trigger. Same ``in_slopes -> out_modes`` contract as
    :class:`~specula.processing_objects.modalrec.Modalrec`, so it is a
    drop-in replacement for it in any existing SCAO YAML.

    This is the counterpart of
    :class:`~iteras.processing_objects.conv2d_net_tester.Conv2dNetTester`
    for actual closed-loop use: no 'labels' input (there is no ground
    truth at runtime), and no evaluation-statistics bookkeeping -- just
    inference, every step.

    The input is the 2D map of the slopes (``Slopes.get2d()``, as Slopes2D
    gives for training) of a single ``in_slopes`` input; ``in_slopes_list``
    is not supported. The network runs on the object's device: on the GPU
    when target_device_idx is a GPU (data exchanged with cupy through
    DLPack, without host copies), on the CPU otherwise.

    Parameters
    ----------
    network_filename : str
        A checkpoint written by Conv2dNetTrainer. The network -- its
        architecture, nmodes, input_channels and n_frames -- is read from
        it, so nothing in the config can disagree with the weights.
        Until n_frames slope maps have arrived, the missing ones repeat
        the first.
    baseline_offset : int, optional
        Index of the first mode within the connected optional
        'baseline' input, if any. When connected, the network's
        (denormalized) output is treated as a residual and the
        baseline is added back to it to form the final out_modes --
        the closed-loop counterpart of Conv2dNetTrainer/Tester's
        residual-learning mode.
    calibrate_gain : bool, optional
        Undo the shrinkage of the network's predictions towards the mean,
        using the per-mode gain measured during training and saved in the
        checkpoint (see cnn_checkpoint.calibration_factor). The loop gains
        in a config then mean what they say: without it, a retrained
        network that shrinks less silently raises the effective loop gain.
        Modes the network barely predicts are left uncorrected, since
        amplifying them would mostly amplify noise.
    """

    def __init__(self,
                 network_filename: str,
                 baseline_offset: int = 0,
                 calibrate_gain: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):
        if TORCH_IMPORT_ERROR is not None:
            raise ImportError(f'{type(self).__name__} needs torch, an optional '
                              f'dependency: pip install torch') from TORCH_IMPORT_ERROR
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.model, stats = load_trained_network(network_filename)
        self.network = network = stats['network']
        self.nmodes = nmodes = network['nmodes']
        self.n_frames = network['n_frames']
        self.baseline_offset = baseline_offset
        self.torch_device = (torch.device('cpu') if self.xp is np
                             else torch.device(f'cuda:{self.target_device_idx}'))
        self.model = self.model.to(self.torch_device)
        self.stacker = FrameStacker(self.n_frames, self.xp)
        self.meanp = stats['meanp']
        self.stdp = stats['stdp']
        self.meanmodes = np.asarray(stats['meanmodes'], dtype=float)
        self.stdmodes = np.asarray(stats['stdmodes'], dtype=float)
        self.calibration = calibration_factor(stats, nmodes) if calibrate_gain else None
        if calibrate_gain and self.calibration is None:
            raise ValueError(f'calibrate_gain is set, but {network_filename} has no per-mode gain '
                             f'measurement (it was trained before this existed): retrain, or set '
                             f'calibrate_gain to false and fold the shrinkage into the loop gains')

        self.inputs['baseline'] = InputValue(type=BaseValue, optional=True)
        self.modes.value = self.xp.zeros(nmodes, dtype=self.dtype)

    @classmethod
    def input_names(cls):
        names = dict(BaseModalrec.input_names())
        names['baseline'] = InputDesc(
            BaseValue, 'Baseline reconstructor output added back to the network '
                      'output for residual learning; see Conv2dNetTrainer (optional)')
        return names

    def setup(self):
        super().setup()
        if self.local_inputs['in_slopes'] is None:
            raise ValueError(f"{type(self).__name__} needs the 'in_slopes' input: in_slopes_list is not supported")
        # Device copies of the output statistics, made here (not in __init__)
        # so that they can still be replaced after construction.
        self._meanmodes = self.to_xp(self.meanmodes, dtype=self.dtype)
        self._stdmodes = self.to_xp(self.stdmodes, dtype=self.dtype)
        self._calibration = (None if self.calibration is None
                             else self.to_xp(self.calibration, dtype=self.dtype))

    def _to_torch(self, x):
        x = self.xp.ascontiguousarray(x, dtype=self.xp.float32)
        if self.xp is np:
            return torch.from_numpy(x)
        return torch.from_dlpack(x)

    def _from_torch(self, t):
        if self.xp is np:
            return t.numpy()
        return self.xp.from_dlpack(t.contiguous())

    def trigger_code(self):
        slopes_obj = self.local_inputs['in_slopes']
        map2d = self.to_xp(slopes_obj.get2d(), dtype=self.xp.float32)
        # The same input as in training: network_input() as Conv2dNetTrainer
        # applies it to the buffered maps
        x = network_input(map2d[self.xp.newaxis, ...], self.network)
        if x.ndim == 3:
            x = x[:, self.xp.newaxis]                       # (1, 1, H, W)
        stacked = self.stacker(x)                           # (1, input_channels * n_frames, H, W)
        ph = (stacked - self.meanp) / self.stdp

        with torch.no_grad():
            preds_normalized = self.model(self._to_torch(ph))

        preds = self._from_torch(preds_normalized[0]) * self._stdmodes + self._meanmodes
        if self._calibration is not None:
            preds = (preds - self._meanmodes) * self._calibration + self._meanmodes

        baseline_in = self.local_inputs['baseline']
        if baseline_in is not None:
            baseline_modes = self.to_xp(baseline_in.get_value())[
                self.baseline_offset:self.baseline_offset + self.nmodes]
            preds = preds + baseline_modes

        self.modes.value[:] = preds
        self.modes.generation_time = self.current_time
