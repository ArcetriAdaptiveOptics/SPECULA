import os

import numpy as np

import iteras
from iteras import cpuArray
from iteras.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from iteras.base_value import BaseValue
from iteras.connections import InputValue
from iteras.lib.frame_stacker import FrameStacker

# torch is optional (the "nn" extra in pyproject.toml): this module must
# import without it; the constructor raises.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from iteras.lib.cnn_checkpoint import (build_network, check_same_network, load_stats, load_weights,
                                           network_input, predict, save_checkpoint, stats_filename)
    from iteras.lib.efficient_u_net import UNetRegressor
    TORCH_IMPORT_ERROR = None
except ImportError as e:
    TORCH_IMPORT_ERROR = e


def to_torch(arr, device='cpu'):
    """numpy/cupy array -> float32 torch tensor on `device` (a cupy array
    goes through DLPack, without a copy to the host)."""
    if iteras.cp is not None and isinstance(arr, iteras.cp.ndarray):
        arr = iteras.cp.ascontiguousarray(arr, dtype=iteras.cp.float32)
        return torch.from_dlpack(arr).to(device=device)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device=device)


class EarlyStopping:
    """Returns True once the loss hasn't improved by more than min_delta for
    `patience` consecutive calls."""
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Conv2dNetTrainer(BaseProcessingObj):
    """
    Trains a UNetRegressor (iteras/lib/efficient_u_net.py) to predict modes
    from 2D maps, on batches buffered by a DataBuffer during the simulation.

    On every trigger, the newest batch is:

    1. read (``input_2d_batch`` -> network input, ``labels`` -> modes to
       predict), with the frames stacked over time (n_frames) and the
       samples taken while the loop wasn't at full gain dropped (optional
       ``gain_mod`` input);
    2. used to set the normalization (from the first batch) or to update the
       output normalization (a slow moving average, see norm_timescale);
    3. split into training samples and validation samples (the most recent
       frames, see val_split); the validation ones join a validation set of
       the most recent val_size;
    4. trained on: epoch_len steps on the whole batch or, with a replay
       buffer (replay_size > 0), minibatches drawn from all the training
       samples collected so far, as many as replay_ratio asks for;
    5. followed by a validation step; every 10 steps the checkpoint is
       saved (see iteras/lib/cnn_checkpoint.py).

    The network works on normalized inputs and outputs, but the loss is
    computed in the labels' physical units (e.g. nm): the network output is
    denormalized with meanmodes/stdmodes first.

    Parameters
    ----------
    network_filename : str
        Checkpoint file (``.pth`` is appended if there's no extension); the
        stats are saved next to it as ``<name>_stats.json``.
    nmodes : int
        Number of modes the network predicts.
    load_from_file : bool
        Resume from network_filename if it exists (it must have been
        trained with the same network parameters).
    channels, depth, conv_block_type, head_type, head_grid, dropout
        Network parameters, see UNetRegressor and the Notes.
    input_channels : int
        Channels of the buffered maps, per frame: 2 for the x/y slope maps
        of a slope computer (its out_slopes_map), 1 for a single map per
        frame (e.g. out_slopes_map of slopes from intensity).
    channel_product : bool
        Use the per-pixel product of the 2 channels of the buffered maps
        (e.g. amplitude and phase) as a single-channel input: requires
        input_channels 1.
    n_frames : int
        Consecutive frames stacked along the channel axis as the network
        input: sample t gets [x(t), x(t-1), ..., x(t-n_frames+1)], so the
        network sees input_channels * n_frames channels. Over a few frames
        the turbulence is strongly correlated while noise isn't.

    label_offset : int
        Index of the first mode within the buffered 'labels' value (e.g. 1
        for labels whose first entry is a placeholder). The labels must have
        at least label_offset + nmodes entries.
    baseline_offset : int
        Index of the first mode within the buffered optional 'baseline'
        input. When it is connected, the network is trained on the residual
        labels - baseline (residual learning: it only has to learn what the
        baseline reconstructor misses).
    max_label_rms : float
        If set, samples whose labels (modes 0..nmodes-1, in quadrature) are
        larger than this are dropped -- see _label_rms_mask(). In the labels'
        units.

    gain_mod_threshold : float
        Samples with gain_mod below this are dropped: while the loop is
        open or at reduced gain the residual is much larger than in closed
        loop, and not representative of it.
    settle_frames : int
        The first settle_frames frames after the loop is back at full gain
        (gain_mod >= 1) count as its re-convergence transient; the count
        carries over across triggers.
    transient_weight : float
        Loss weight, in [0, 1], of those transient frames (1: same as the
        others; 0: dropped).

    norm_timescale : float
        Time constant, in seconds of simulated time, of the output
        statistics' moving average. It should span several independent
        draws of the conditions; getting it somewhat wrong costs little,
        since the loss is in physical units and the predictions are kept
        unchanged by each update.

    val_split : float
        Fraction of each batch used for validation: its most recent frames,
        as one contiguous block. Frames a few ms apart are nearly identical,
        so with a random split every validation frame would sit next to
        training frames, and the validation loss would reward memorizing them
        rather than predicting new data. The step line reports the loss and
        the FVU (error variance / label variance) on them; the loss, in
        physical units, follows the conditions of the last few batches, the
        FVU much less, so progress, the best step and early stopping are
        judged by the FVU.
    val_size : int
        Validation samples kept, the most recent ones. It sets how much of
        the run the validation FVU averages over: with the conditions
        changing every few seconds, a window of a few seconds makes it follow
        the conditions rather than the network. It only affects what is
        reported, not the training.
    epoch_len : int
        Full-batch steps per trigger, without a replay buffer.
    replay_size : int
        If > 0, keep up to this many training samples (a ring buffer in CPU
        RAM: replay_size x (input pixels + nmodes) x 4 bytes) and train on
        minibatches drawn from all of them. Each batch covers a fraction of a
        second of one atmospheric condition: trained on alone, the network
        forgets earlier conditions.
    replay_subsample : int
        Only every Nth training frame enters the replay buffer: consecutive
        frames at kHz rates are nearly identical, and a larger N makes the
        buffer span more time for the same memory.
    replay_ratio : float
        Minibatch draws per new sample: every trigger takes
        replay_ratio * (samples it added) / replay_batch minibatch steps,
        drawn from the whole buffer. Once the buffer is full, each sample is
        thus drawn replay_ratio times on average while it stays in it. Tying
        the steps to the new data, rather than fixing them, bounds the passes
        over the buffer: with a fixed number of steps, fewer samples per
        trigger -- a larger replay_subsample, or a batch mostly filtered
        out -- meant many more passes over the same samples, and the buffer
        got memorized.
    replay_batch : int
        Minibatch size.
    loss_delta : float
        Huber loss transition point, in the labels' units (the default suits
        modes in nm): errors above it count linearly instead of
        quadratically, so occasional large-residual samples don't dominate
        the loss.
    norm_loss_weight : float
        Weight of an extra loss term: the squared error per mode normalized
        by that mode's std (stdmodes). In the physical-unit loss a mode counts
        in proportion to its variance, so small modes get little push to be
        learned; this term weighs every mode by its own spread. The reported
        losses don't include it.
    grad_clip_value : float
        Maximum total gradient norm; None or <= 0 disables clipping.
    lr_decay : float
        If set, in (0, 1): the learning rate (initially 1e-3) is multiplied
        by it after every trigger (never below 5e-6). Default None: reduced
        by 0.8 when the validation FVU stops improving for patience // 10
        triggers. Either way the learning rate is saved in the checkpoint and
        a resumed run continues from it, so the annealing spans runs.
    patience : int
        Training stops after this many triggers without improvement of the
        validation loss.

    mode_gain_alpha : float
        Weight of the newest batch in the moving average of the per-mode
        prediction gain (see _update_mode_gain), saved in the checkpoint and
        used by calibrate_gain. The default follows the latest conditions
        within ~10 triggers; when a checkpoint is trained over runs in
        different conditions (e.g. different magnitudes), a small value
        averages over several runs instead of reflecting only the last one.

    data_parallel : bool
        Train on all visible GPUs with torch's DataParallel instead of the
        object's GPU only. Off by default: on a GPU shared with other
        simulations it would take all of them.

    Notes
    -----
    Network parameters: channels, depth, conv_block_type, head_type, head_grid, dropout
    -- see UNetRegressor (channels = base_channels, dropout = dropout_level).
    They are saved in the checkpoint together with nmodes, input_channels,
    channel_product and n_frames, and Conv2dNetTester/Conv2dNetRec read them
    from there. The maps must be at least 2 ** depth pixels wide; for small
    maps with the 'spatial' head, head_grid should be about the map size.

    Loop transients: the optional 'gain_mod' input is the loop gain
    modulation (see gain_mod_threshold, settle_frames, transient_weight).

    Normalization. The input is normalized by one mean/std over all pixels,
    taken from the first batch and then kept: the network's first block ends
    in a GroupNorm, which rescales every sample itself, so the network is
    insensitive to that choice and nothing is gained by tracking it. The
    output is normalized per mode, and those statistics do matter -- the
    network predicts in that space -- but a few seconds of data don't
    represent an atmosphere that keeps changing. So they follow a slow moving
    average, and every update rescales the network's output layers so that
    its denormalized predictions stay exactly the same: the statistics can
    move at any rate without the network having to chase them.

    Device: the network trains on the object's device, the GPU of
    target_device_idx, or the CPU when the object is on the CPU.
    """

    def __init__(self,
                 network_filename: str,
                 nmodes: int = 20,
                 load_from_file: bool = False,
                 channels: int = 32,
                 depth: int = 5,
                 conv_block_type: int = 0,
                 head_type: str = 'pooled',
                 head_grid: int = 32,
                 dropout: float = 0.01,
                 input_channels: int = 2,
                 channel_product: bool = False,
                 n_frames: int = 1,
                 label_offset: int = 0,
                 baseline_offset: int = 0,
                 max_label_rms: float = None,
                 gain_mod_threshold: float = 1.0,
                 settle_frames: int = 0,
                 transient_weight: float = 1.0,
                 norm_timescale: float = 30.0,
                 val_split: float = 0.2,
                 val_size: int = 1000,
                 epoch_len: int = 20,
                 replay_size: int = 0,
                 replay_subsample: int = 1,
                 replay_ratio: float = 20.0,
                 replay_batch: int = 64,
                 loss_delta: float = 20.0,
                 norm_loss_weight: float = 0.0,
                 grad_clip_value: float = 1.0,
                 lr_decay: float = None,
                 patience: int = 600,
                 mode_gain_alpha: float = 0.3,
                 data_parallel: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):
        if TORCH_IMPORT_ERROR is not None:
            raise ImportError(f'{type(self).__name__} needs torch, an optional '
                              f'dependency: pip install torch') from TORCH_IMPORT_ERROR
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if not norm_timescale > 0:
            raise ValueError(f'norm_timescale must be > 0, got {norm_timescale}')
        if not loss_delta > 0:
            raise ValueError(f'loss_delta must be > 0, got {loss_delta}')
        if replay_size < 0 or replay_subsample < 1 or replay_batch < 1 or not replay_ratio > 0:
            raise ValueError('replay_size must be >= 0, replay_subsample and replay_batch >= 1, '
                             'and replay_ratio > 0')
        if lr_decay is not None and not 0 < lr_decay < 1:
            raise ValueError(f'lr_decay must be in (0, 1) or None, got {lr_decay}')
        if settle_frames < 0 or not 0 <= transient_weight <= 1:
            raise ValueError('settle_frames must be >= 0 and transient_weight in [0, 1]')
        if val_size < 1:
            raise ValueError(f'val_size must be >= 1, got {val_size}')
        if n_frames < 1 or norm_loss_weight < 0:
            raise ValueError('n_frames must be >= 1 and norm_loss_weight >= 0')
        if channel_product and input_channels != 1:
            raise ValueError('channel_product makes a single-channel input: it needs input_channels 1')

        if not os.path.splitext(network_filename)[1]:
            network_filename += '.pth'
        self.network_filename = network_filename
        self.stats_filename = stats_filename(network_filename)

        self.nmodes = nmodes
        self.input_channels = input_channels
        self.n_frames = n_frames
        # everything needed to rebuild the network, saved in the checkpoint
        self.network = dict(nmodes=nmodes, input_channels=input_channels, n_frames=n_frames,
                            channels=channels, depth=depth, conv_block_type=conv_block_type,
                            head_type=head_type, head_grid=head_grid, dropout=dropout,
                            channel_product=channel_product)
        self.label_offset = label_offset
        self.baseline_offset = baseline_offset
        self.max_label_rms = max_label_rms
        self.gain_mod_threshold = gain_mod_threshold
        self.settle_frames = settle_frames
        self.transient_weight = transient_weight
        self.norm_timescale = norm_timescale
        self.val_split = val_split
        self.epoch_len = epoch_len
        self.replay_size = replay_size
        self.replay_subsample = replay_subsample
        self.replay_ratio = replay_ratio
        self._replay_new = 0        # samples the last _replay_add put in the buffer
        self.replay_batch = replay_batch
        self.loss_delta = loss_delta
        self.norm_loss_weight = norm_loss_weight
        # float('inf'): clip_grad_norm_ then only measures the norm.
        self.grad_clip_value = grad_clip_value if grad_clip_value and grad_clip_value > 0 else float('inf')
        self.lr_decay = lr_decay
        self.min_lr = 5e-6

        self.stacker = FrameStacker(n_frames, self.xp)
        self._frames_at_full_gain = 0
        self._warned_no_gain_mod = False
        self._empty_batches = 0
        self._checked_input_size = False

        # Normalization: restored from the stats file when resuming, otherwise
        # set from the first batch (see trigger()). The output statistics are
        # kept as first and second moments, which is what a moving average of
        # a variance has to average.
        self.meanp = self.stdp = self.meanmodes = self.stdmodes = None
        self._out_m1 = self._out_m2 = None
        self._norm_time = None
        self.mode_gain = None       # per-mode prediction gain, see _update_mode_gain
        if not 0 < mode_gain_alpha <= 1:
            raise ValueError(f'mode_gain_alpha must be in (0, 1], got {mode_gain_alpha}')
        self.mode_gain_alpha = mode_gain_alpha
        self._resume_lr = None      # learning rate saved in the checkpoint, if resuming

        model = build_network(self.network)
        if load_from_file:
            self._load(model)

        if self.xp is np or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{self.target_device_idx}')
        n_gpus = torch.cuda.device_count()
        if data_parallel and self.device.type == 'cuda' and n_gpus > 1:
            # DataParallel wants the model on device_ids[0]: the object's GPU first
            t = self.device.index
            device_ids = [t] + [i for i in range(n_gpus) if i != t]
            model = nn.DataParallel(model.to(self.device), device_ids=device_ids)
            self.logger.info(f'Training on {n_gpus} GPUs (DataParallel)')
        else:
            self.logger.info(f'Training on {self.device}')
        self.model = model.to(self.device)

        # Plain Huber loss in physical units, used for the reported losses;
        # training uses _loss(), which can add weights and the normalized term.
        self.loss_fn = nn.HuberLoss(delta=loss_delta)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._resume_lr or 1e-3)
        if self._resume_lr is not None:
            self.logger.info(f'Learning rate resumed at {self._resume_lr:.2e}')
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=patience // 10, min_lr=self.min_lr)
        self.early_stopping = EarlyStopping(patience=patience)
        self.should_stop = False

        self.inputs['input_2d_batch'] = InputValue(type=BaseValue)
        self.inputs['labels'] = InputValue(type=BaseValue)
        self.inputs['baseline'] = InputValue(type=BaseValue, optional=True)
        self.inputs['gain_mod'] = InputValue(type=BaseValue, optional=True)
        self.outputs['loss'] = BaseValue(target_device_idx=target_device_idx)
        self.outputs['val_loss'] = BaseValue(target_device_idx=target_device_idx)

        self.max_val = val_size
        self.val_inputs = self.val_targets = None
        self.replay_x = self.replay_y = self.replay_w = None
        self.replay_count = 0
        self._replay_pos = 0
        self.step_count = 0
        self.loss = self.val_loss = None
        self.val_fvu = None
        self.best_val_fvu = float('inf')

    @classmethod
    def input_names(cls):
        return {
            'input_2d_batch': InputDesc(BaseValue, 'Buffered 2D maps, (B, H, W) or (B, C, H, W), '
                                                   'e.g. a Slopec out_slopes_map through a DataBuffer'),
            'labels': InputDesc(BaseValue, 'Buffered modes to predict, (B, >= label_offset + nmodes)'),
            'baseline': InputDesc(BaseValue, 'Buffered baseline modes: the network learns '
                                             'labels - baseline (optional)'),
            'gain_mod': InputDesc(BaseValue, 'Buffered loop gain modulation: samples below '
                                             'gain_mod_threshold are dropped (optional)'),
        }

    @classmethod
    def output_names(cls):
        return {
            'loss': OutputDesc(BaseValue, 'Training loss of the last trigger (Huber, labels units)'),
            'val_loss': OutputDesc(BaseValue, 'Validation loss of the last trigger (Huber, labels units)'),
        }

    # ------------------------------------------------------------------
    #   Checkpoint
    # ------------------------------------------------------------------

    def _load(self, model):
        if not os.path.isfile(self.network_filename):
            self.logger.info(f'No checkpoint at {self.network_filename}: training a new network')
            return
        # Checked before loading: a mismatch must stop here, not end with a
        # new network overwriting the checkpoint.
        check_same_network(self.network_filename, self.network)
        load_weights(model, self.network_filename)
        self.logger.info(f'Model loaded from {self.network_filename}')
        try:
            stats = load_stats(self.network_filename)
        except FileNotFoundError:
            self.logger.warning('stats file not found: the normalization will be '
                                'estimated again from the data')
            return
        self.meanp = stats['meanp']
        self.stdp = stats['stdp']
        self._set_output_moments(np.asarray(stats['meanmodes'], dtype=np.float64),
                                 np.asarray(stats['stdmodes'], dtype=np.float64) ** 2
                                 + np.asarray(stats['meanmodes'], dtype=np.float64) ** 2)
        self.mode_gain = np.asarray(stats['mode_gain']) if stats.get('mode_gain') else None
        self._resume_lr = stats.get('lr')

    def _inner_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _save(self, val_loss):
        save_checkpoint(self._inner_model(), self.network_filename, {
            'network': self.network,
            'meanp': float(self.meanp),
            'stdp': float(self.stdp),
            'meanmodes': cpuArray(self.meanmodes).tolist(),
            'stdmodes': cpuArray(self.stdmodes).tolist(),
            'mode_gain': self.mode_gain.tolist() if self.mode_gain is not None else None,
            'min_loss': val_loss,   # this save's validation loss (not the best ever)
            'val_fvu': self.val_fvu,
            'lr': self.optimizer.param_groups[0]['lr'],
        })
        self.logger.info(f'Model saved to {self.network_filename}')

    # ------------------------------------------------------------------
    #   Data
    # ------------------------------------------------------------------

    def _read_batch(self):
        """The newest buffered batch, without the samples dropped because of
        gain_mod or max_label_rms: (network inputs, not normalized yet; modes
        to predict; frames since the loop came back to full gain, or None;
        loss weights, or None). None if there is nothing to train on."""
        x_in = self.local_inputs['input_2d_batch']
        labels_in = self.local_inputs['labels']
        if x_in is None or labels_in is None:
            return None
        # Stacked before any frame is dropped: each sample keeps its true
        # preceding frames.
        x = network_input(x_in.get_value(), self.network)
        if not self._checked_input_size:
            UNetRegressor.check_input_size(x.shape[-2:], self.network['depth'])
            self._checked_input_size = True
        x = self.stacker(x)
        labels = labels_in.get_value()
        first = self.label_offset
        if labels.shape[1] < first + self.nmodes:
            raise ValueError(f"the labels have {labels.shape[1]} entries, fewer than label_offset + nmodes = "
                             f"{first} + {self.nmodes}: check label_offset and nmodes")
        modes = labels[:, first:first + self.nmodes]
        baseline_in = self.local_inputs['baseline']
        if baseline_in is not None:
            b = self.baseline_offset
            modes = modes - baseline_in.get_value()[:, b:b + self.nmodes]

        fsr = None
        keep = self._label_rms_mask(modes)
        gain_mod_in = self.local_inputs['gain_mod']
        if gain_mod_in is None:
            if self.settle_frames > 0 and not self._warned_no_gain_mod:
                self.logger.warning("settle_frames needs the 'gain_mod' input, "
                                    "which isn't connected: ignored")
                self._warned_no_gain_mod = True
        else:
            gain_mod = self.xp.asarray(gain_mod_in.get_value()).reshape(-1)
            fsr = self._frames_since_reclose(gain_mod)
            gain_keep = gain_mod >= self.gain_mod_threshold
            if self.settle_frames > 0 and self.transient_weight == 0:
                gain_keep = gain_keep & self.xp.asarray(fsr > self.settle_frames)
            keep = gain_keep if keep is None else keep & gain_keep

        if keep is None:
            self._empty_batches = 0
            return x, modes, None, None
        if not self.xp.any(keep):
            self._empty_batches += 1
            if self._empty_batches == 3:
                self.logger.warning('3 batches in a row with no usable sample. If this continues, '
                                    'the loop has most likely diverged: training is idle from here on')
            return None
        self._empty_batches = 0
        if fsr is not None:
            fsr = fsr[cpuArray(keep)]
        weights = None
        if fsr is not None and self.settle_frames > 0 and 0 < self.transient_weight < 1:
            weights = np.where(fsr <= self.settle_frames, self.transient_weight, 1.0).astype(np.float32)
        return x[keep], modes[keep], fsr, weights

    def _label_rms_mask(self, modes):
        """Which samples to keep by the size of their labels, or None when
        max_label_rms is off. Samples whose residual is far larger than usual
        are not representative of the loop this network is for: at that size a
        non-modulated pyramid is outside its linear range, so there is no
        input -> mode relation left to learn (a linear map fitted on such
        batches fails on the next one just as badly), and a diverging loop
        would otherwise fill the replay buffer with them."""
        if self.max_label_rms is None:
            return None
        rms = self.xp.sqrt(self.xp.sum(modes ** 2, axis=1))
        keep = rms <= self.max_label_rms
        dropped = int(modes.shape[0] - self.xp.sum(keep))
        if dropped > 0.1 * modes.shape[0]:
            self.logger.info(f'{dropped}/{modes.shape[0]} samples dropped: label RMS above '
                             f'max_label_rms={self.max_label_rms:g} '
                             f'(batch median {float(self.xp.median(rms)):.0f})')
        return keep

    def _frames_since_reclose(self, gain_mod):
        """For each frame (in time order), how many consecutive frames the
        loop has been at full gain (gain_mod >= 1) up to and including it;
        0 while it isn't. Carries over from the previous trigger."""
        g = cpuArray(gain_mod).reshape(-1)
        out = np.empty(g.shape[0], dtype=np.int64)
        count = self._frames_at_full_gain
        for i, value in enumerate(g):
            count = count + 1 if value >= 1.0 - 1e-6 else 0
            out[i] = count
        self._frames_at_full_gain = count
        return out

    # ------------------------------------------------------------------
    #   Normalization
    # ------------------------------------------------------------------

    def _init_normalization(self, x, modes):
        """From the first batch: the input normalization, kept from then on,
        and the starting point of the output's moving average."""
        xx = to_torch(x).to(torch.float64)
        self.meanp = float(xx.mean())
        self.stdp = float(xx.std(unbiased=False)) + 1e-8
        y = cpuArray(modes).astype(np.float64)
        self._set_output_moments(y.mean(0), (y ** 2).mean(0))
        self._norm_time = self.t_to_seconds(self.current_time)
        self.logger.info(f'Normalization initialized from the first {x.shape[0]} samples; the '
                         f'output statistics follow a {self.norm_timescale:g} s moving average from here')

    def _set_output_moments(self, m1, m2):
        self._out_m1, self._out_m2 = m1, m2
        std = np.sqrt(np.maximum(m2 - m1 ** 2, 0.0)) + 1e-8
        self.meanmodes = self.xp.asarray(m1, dtype=self.dtype)
        self.stdmodes = self.xp.asarray(std, dtype=self.dtype)

    def _update_output_normalization(self, modes):
        """Move the output statistics towards this batch's, with the weight a
        norm_timescale moving average gives the simulated time since the
        last update, and rescale the output layers to match."""
        now = self.t_to_seconds(self.current_time)
        if self._norm_time is None:      # first batch after resuming: no interval yet
            self._norm_time = now
            return
        alpha = 1.0 - np.exp(-max(now - self._norm_time, 0.0) / self.norm_timescale)
        self._norm_time = now
        y = cpuArray(modes).astype(np.float64)
        old_mean = cpuArray(self.meanmodes).astype(np.float64)
        old_std = cpuArray(self.stdmodes).astype(np.float64)
        self._set_output_moments((1 - alpha) * self._out_m1 + alpha * y.mean(0),
                                 (1 - alpha) * self._out_m2 + alpha * (y ** 2).mean(0))
        self._preserve_outputs(old_mean, old_std, cpuArray(self.meanmodes).astype(np.float64),
                               cpuArray(self.stdmodes).astype(np.float64))

    def _preserve_outputs(self, old_mean, old_std, new_mean, new_std):
        """Rescale the network's output layers so that its denormalized
        predictions are unchanged by a new output normalization. A
        prediction is std * o + mean, with o a sum of linear layers, so
        o' = (std / std') * o + (mean - mean') / std' is exact: scale every
        output layer's weights and bias by std / std', and add the shift to
        one bias."""
        scale = to_torch(old_std / new_std, self.device)
        shift = to_torch((old_mean - new_mean) / new_std, self.device)
        with torch.no_grad():
            for i, layer in enumerate(self._inner_model().output_layers()):
                layer.weight.mul_(scale[:, None])
                layer.bias.mul_(scale)
                if i == 0:
                    layer.bias.add_(shift)

    # ------------------------------------------------------------------
    #   Replay buffer
    # ------------------------------------------------------------------

    def _replay_add(self, x, y, w=None):
        """Add samples (CPU tensors, rows in time order: network inputs before
        normalization, labels in physical units and, optionally, loss
        weights) to the ring buffer, keeping every replay_subsample-th one."""
        if w is None:
            w = torch.ones(x.shape[0])
        step = self.replay_subsample
        x, y, w = x[::step][-self.replay_size:], y[::step][-self.replay_size:], w[::step][-self.replay_size:]
        self._replay_new = x.shape[0]
        if x.shape[0] == 0:
            return
        if self.replay_x is None:
            self.replay_x = torch.empty((self.replay_size,) + tuple(x.shape[1:]), dtype=torch.float32)
            self.replay_y = torch.empty((self.replay_size, y.shape[1]), dtype=torch.float32)
            self.replay_w = torch.empty(self.replay_size, dtype=torch.float32)
        n = x.shape[0]
        idx = (self._replay_pos + torch.arange(n)) % self.replay_size
        self.replay_x[idx] = x.to(torch.float32)
        self.replay_y[idx] = y.to(torch.float32)
        self.replay_w[idx] = w.to(torch.float32)
        self._replay_pos = (self._replay_pos + n) % self.replay_size
        self.replay_count = min(self.replay_count + n, self.replay_size)

    def _replay_steps(self):
        """Minibatch steps for this trigger: replay_ratio draws per sample the
        trigger added, at least one."""
        return max(1, int(round(self.replay_ratio * self._replay_new / self.replay_batch)))

    def _replay_minibatch(self):
        """A uniformly drawn minibatch, with the inputs normalized with the
        *current* statistics: (inputs, labels, loss weights)."""
        idx = torch.randint(0, self.replay_count, (min(self.replay_batch, self.replay_count),))
        x = (self.replay_x[idx].to(self.device) - float(self.meanp)) / float(self.stdp)
        if x.dim() == 3:    # single-channel frames stored without a channel axis
            x = x.unsqueeze(1)
        return x, self.replay_y[idx].to(self.device), self.replay_w[idx].to(self.device)

    # ------------------------------------------------------------------
    #   Training
    # ------------------------------------------------------------------

    def trigger(self):
        if self.should_stop:
            return
        batch = self._read_batch()
        if batch is None:
            return
        x, modes, fsr, weights = batch

        if self.meanmodes is None:
            self._init_normalization(x, modes)
        else:
            self._update_output_normalization(modes)

        x_norm = (x - self.meanp) / self.stdp
        if x_norm.ndim == 3:    # single-channel frames: add the channel axis
            x_norm = x_norm[:, self.xp.newaxis]

        # Validation: the batch's most recent frames, as one block (see val_split)
        n = x.shape[0]
        n_val = max(1, int(n * self.val_split))
        train_idx, val_idx = np.arange(n - n_val), np.arange(n - n_val, n)

        dev = self.device
        inputs = to_torch(x_norm[train_idx], dev)
        targets = to_torch(modes[train_idx], dev)
        train_w = torch.from_numpy(weights[train_idx]).to(dev) if weights is not None else None
        mean_t = to_torch(self.meanmodes, dev)
        std_t = to_torch(self.stdmodes, dev)
        self._add_validation(to_torch(x_norm[val_idx], dev), to_torch(modes[val_idx], dev))

        if self.replay_size > 0:
            self._replay_add(to_torch(x[train_idx]), to_torch(modes[train_idx]),
                             torch.from_numpy(weights[train_idx]) if weights is not None else None)

        loss = self._train(inputs, targets, train_w, mean_t, std_t)
        if loss is None:
            return
        self.loss = loss
        self.step_count += 1
        if self.lr_decay is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = max(group['lr'] * self.lr_decay, self.min_lr)

        self._measure_mode_gain(inputs, targets, mean_t, std_t)
        self._validate(mean_t, std_t)

    def _add_validation(self, inputs, targets):
        """Append to the validation set, keeping the most recent max_val samples."""
        if self.val_inputs is None:
            self.val_inputs, self.val_targets = inputs, targets
        else:
            self.val_inputs = torch.cat([self.val_inputs, inputs])[-self.max_val:]
            self.val_targets = torch.cat([self.val_targets, targets])[-self.max_val:]

    def _loss(self, preds, targets, weights, std_t):
        """Training loss, in physical units: the Huber loss per sample (mean
        over modes), plus norm_loss_weight times the squared error
        normalized per mode by std_t; then the mean over samples, weighted
        by `weights` if given."""
        per_sample = F.huber_loss(preds, targets, reduction='none', delta=self.loss_delta).mean(dim=1)
        if self.norm_loss_weight > 0:
            per_sample = per_sample + self.norm_loss_weight * (((preds - targets) / std_t) ** 2).mean(dim=1)
        if weights is None:
            return per_sample.mean()
        return (weights * per_sample).sum() / weights.sum()

    def _train(self, inputs, targets, weights, mean_t, std_t):
        """This trigger's gradient steps: minibatches from the replay buffer
        (see replay_ratio) or, without one, epoch_len steps on the new batch.
        Returns the training loss (plain Huber loss in physical units: the
        mean over the replay steps, or the last full-batch step), or None if
        the loss is not finite."""
        self.model.train()
        use_replay = self.replay_size > 0
        losses = []
        n_steps = self._replay_steps() if use_replay else self.epoch_len
        for _ in range(n_steps):
            x, y, w = self._replay_minibatch() if use_replay else (inputs, targets, weights)
            self.optimizer.zero_grad()
            preds = self.model(x) * std_t + mean_t      # denormalized: physical units
            loss = self._loss(preds, y, w, std_t)
            if not torch.isfinite(loss):
                self.logger.warning('NaN/Inf loss, skipping this batch')
                return None
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            self.optimizer.step()
            losses.append(self.loss_fn(preds.detach(), y).item())
        return float(np.mean(losses)) if use_replay else losses[-1]

    def _validate(self, mean_t, std_t):
        preds = predict(self.model, self.val_inputs, mean_t, std_t)
        t = self.val_targets
        self.val_loss = self.loss_fn(preds, t).item()
        self.val_fvu = float(((preds - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum().clamp_min(1e-30))
        if self.lr_decay is None:
            self.plateau_scheduler.step(self.val_fvu)
        self.best_val_fvu = min(self.best_val_fvu, self.val_fvu)

        # Periodic, not only on improvement: otherwise a run whose validation
        # loss never beats an early best step would save nothing to resume from.
        if self.step_count % 10 == 0:
            self._save(self.val_loss)

        lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(f'Step {self.step_count} | LR {lr:.2e} | Train {self.loss:.6f} | '
                         f'Val {self.val_loss:.6f} (FVU {self.val_fvu:.3f}) | Best FVU {self.best_val_fvu:.3f}')

        if self.early_stopping(self.val_fvu):
            self.should_stop = True
            self.logger.info(f'Early stopping triggered at step {self.step_count}')

    def _measure_mode_gain(self, inputs, targets, mean_t, std_t):
        """The prediction gain (see _update_mode_gain) on the newest batch's
        training frames, after training on them. In-sample on purpose: on
        held-out frames the network shrinks the modes it predicts poorly much
        harder (gains of 0.2-0.4 early in training), and dividing by that
        would mostly amplify noise -- at a gain of 0.3 and FVU 0.75, the
        calibrated prediction's noise is ~1.7x the signal. The closed-loop
        gains were tuned with this measurement."""
        self._update_mode_gain(predict(self.model, inputs, mean_t, std_t), targets)

    def _update_mode_gain(self, preds, targets):
        """Per-mode prediction gain cov(pred, true) / var(true): how much the
        network shrinks each mode towards the mean, which in a loop acts
        exactly like a lower gain on that mode. It is saved in the stats file
        so Conv2dNetRec / Conv2dNetTester can undo it (see
        cnn_checkpoint.calibration_factor), which keeps the loop gains in a
        config meaningful across retrains -- a network that shrinks less
        after a retrain otherwise silently raises the effective loop gain.
        Smoothed over triggers, since one batch is noisy."""
        p = preds.detach().to(torch.float64)
        t = targets.to(torch.float64)
        centred = t - t.mean(0)
        gain = ((centred * (p - p.mean(0))).mean(0)
                / centred.pow(2).mean(0).clamp_min(1e-30)).cpu().numpy()
        alpha = self.mode_gain_alpha
        self.mode_gain = gain if self.mode_gain is None else (1 - alpha) * self.mode_gain + alpha * gain

    # ------------------------------------------------------------------

    def post_trigger(self):
        super().post_trigger()
        # generation_time must follow, or downstream consumers (e.g. a
        # PlotDisplay) never see these outputs as new.
        for name, value in (('loss', self.loss), ('val_loss', self.val_loss)):
            if value is not None:
                self.outputs[name].set_value(value)
                self.outputs[name].generation_time = self.current_time

    def finalize(self):
        self.logger.info(f'Training complete: {self.step_count} steps, best validation FVU '
                         f'{self.best_val_fvu:.3f}, stats saved to {self.stats_filename}')
        super().finalize()
