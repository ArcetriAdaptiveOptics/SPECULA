
from iteras import cpuArray, np
from iteras.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from iteras.base_value import BaseValue
from iteras.connections import InputValue
from iteras.lib.frame_stacker import FrameStacker

# torch is optional (the "nn" extra in pyproject.toml): this module must
# import without it; the constructor raises.
try:
    import torch
    from iteras.lib.cnn_checkpoint import calibration_factor, load_trained_network, network_input
    from iteras.processing_objects.conv2d_net_trainer import to_torch
    TORCH_IMPORT_ERROR = None
except ImportError as e:
    TORCH_IMPORT_ERROR = e


def mode_groups(nmodes):
    '''Mode index ranges [start, end) of the per-group statistics'''
    edges = [e for e in (0, 5, 20, 50, 100, 200, 500) if e < nmodes] + [nmodes]
    return list(zip(edges[:-1], edges[1:]))


class Conv2dNetTester(BaseProcessingObj):
    """
    Evaluates a network trained by Conv2dNetTrainer on buffered batches
    with known modes: no training, just the predictions ('prediction'
    output, and the batch's mean squared error as 'loss') and, at the end
    of the simulation, per-mode error statistics.

    The network -- its architecture, nmodes, input_channels and n_frames --
    is read from the checkpoint; label_offset / baseline_offset must match
    the ones used for training (see Conv2dNetTrainer).

    With the optional 'baseline' input connected, the network output is a
    residual: the baseline is added back to it before comparing with the
    labels. With the optional 'gain_mod' input connected, samples with
    gain_mod < gain_mod_threshold (loop open or at reduced gain, not
    representative of closed-loop operation) are left out of the statistics.

    With the optional 'reference' input connected -- another estimate of the
    same modes, typically the linear reconstructor's out_modes buffered
    alongside the labels -- its error is accumulated on exactly the same
    frames and reported next to the network's, which is the way to tell
    whether the network reconstructs better than the reconstructor it is
    meant to replace.

    The network runs on the object's device. The statistics are accumulated
    as per-mode sums, so the memory does not grow with the test length.

    Parameters
    ----------
    network_filename : str
        A checkpoint written by Conv2dNetTrainer.
    calibrate_gain : bool
        Undo the shrinkage of the predictions measured during training, as
        Conv2dNetRec does (see cnn_checkpoint.calibration_factor).
    label_offset, baseline_offset, reference_offset : int
        Index of the first mode within the labels, baseline and reference inputs.
    gain_mod_threshold : float
        Samples with gain_mod below this are left out.
    """

    def __init__(self,
                 network_filename: str,
                 calibrate_gain: bool = False,
                 label_offset: int = 0,
                 baseline_offset: int = 0,
                 reference_offset: int = 0,
                 gain_mod_threshold: float = 1.0,
                 target_device_idx: int = None,
                 precision: int = None):
        if TORCH_IMPORT_ERROR is not None:
            raise ImportError(f'{type(self).__name__} needs torch, an optional '
                              f'dependency: pip install torch') from TORCH_IMPORT_ERROR
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.model, stats = load_trained_network(network_filename)
        self.network = network = stats['network']
        self.nmodes = nmodes = network['nmodes']
        self.torch_device = (torch.device('cpu') if self.xp is np
                             else torch.device(f'cuda:{self.target_device_idx}'))
        self.model = self.model.to(self.torch_device)
        self.label_offset = label_offset
        self.baseline_offset = baseline_offset
        self.reference_offset = reference_offset
        self.gain_mod_threshold = gain_mod_threshold
        self.stacker = FrameStacker(network['n_frames'], self.xp)
        self.meanp = stats['meanp']
        self.stdp = stats['stdp']
        self.meanmodes = self.xp.array(stats['meanmodes'])
        self.stdmodes = self.xp.array(stats['stdmodes'])
        # see Conv2dNetRec for what this is; scored here so the test matches
        # how the network is used in closed loop
        factor = calibration_factor(stats, nmodes) if calibrate_gain else None
        self.calibration = self.xp.array(factor) if factor is not None else None
        if calibrate_gain and self.calibration is None:
            raise ValueError(f'calibrate_gain is set, but {network_filename} has no per-mode gain '
                             f'measurement (it was trained before this existed)')

        self.inputs['input_2d_batch'] = InputValue(type=BaseValue)
        self.inputs['labels'] = InputValue(type=BaseValue)
        self.inputs['baseline'] = InputValue(type=BaseValue, optional=True)
        self.inputs['gain_mod'] = InputValue(type=BaseValue, optional=True)
        self.inputs['reference'] = InputValue(type=BaseValue, optional=True)
        self.outputs['loss'] = BaseValue(target_device_idx=target_device_idx)
        self.outputs['prediction'] = BaseValue(target_device_idx=target_device_idx)

        # Per-mode sums over the samples: errors, |errors|, errors^2, SMAPE
        # terms, labels, labels^2 and, on the frames with a reference, its errors^2
        self.count = 0
        self.reference_count = 0
        self.sums = {k: self.xp.zeros(nmodes) for k in ('e', 'abs_e', 'e2', 'smape', 't', 't2', 'ref_e2')}

    @classmethod
    def input_names(cls):
        return {
            'input_2d_batch': InputDesc(BaseValue, 'Buffered 2D maps, as for Conv2dNetTrainer'),
            'labels': InputDesc(BaseValue, 'Buffered true modes'),
            'baseline': InputDesc(BaseValue, 'Buffered baseline modes added back to the network '
                                             'output (optional)'),
            'gain_mod': InputDesc(BaseValue, 'Buffered loop gain modulation: samples below '
                                             'gain_mod_threshold are left out (optional)'),
            'reference': InputDesc(BaseValue, 'Buffered modes of another estimator, scored on '
                                              'the same frames (optional)'),
        }

    @classmethod
    def output_names(cls):
        return {
            'loss': OutputDesc(BaseValue, 'Mean squared error of the last batch'),
            'prediction': OutputDesc(BaseValue, 'Predicted modes of the last batch'),
        }

    def trigger(self):
        x_in = self.local_inputs['input_2d_batch']
        labels_in = self.local_inputs['labels']
        if x_in is None or labels_in is None:
            return

        # Stacked before any frame is dropped (see Conv2dNetTrainer).
        x = self.stacker(network_input(x_in.get_value(), self.network))
        first = self.label_offset
        labels = labels_in.get_value()
        if labels.shape[1] < first + self.nmodes:
            raise ValueError(f"the labels have {labels.shape[1]} entries, fewer than label_offset + nmodes = "
                             f"{first} + {self.nmodes}: check label_offset")
        modes = labels[:, first:first + self.nmodes]
        baseline = None
        baseline_in = self.local_inputs['baseline']
        if baseline_in is not None:
            b = self.baseline_offset
            baseline = baseline_in.get_value()[:, b:b + self.nmodes]
        reference = None
        reference_in = self.local_inputs['reference']
        if reference_in is not None:
            r = self.reference_offset
            reference = reference_in.get_value()[:, r:r + self.nmodes]

        gain_mod_in = self.local_inputs['gain_mod']
        if gain_mod_in is not None:
            keep = self.xp.asarray(gain_mod_in.get_value()).reshape(-1) >= self.gain_mod_threshold
            if not self.xp.any(keep):
                return
            x, modes = x[keep], modes[keep]
            if baseline is not None:
                baseline = baseline[keep]
            if reference is not None:
                reference = reference[keep]

        x = (x - self.meanp) / self.stdp
        if x.ndim == 3:     # single-channel frames: add the channel axis
            x = x[:, self.xp.newaxis]
        with torch.no_grad():
            preds_normalized = self.model(to_torch(x, self.torch_device))
        if self.xp is np:
            preds_normalized = preds_normalized.cpu().numpy()
        else:
            preds_normalized = self.xp.from_dlpack(preds_normalized.contiguous())
        preds = preds_normalized * self.stdmodes + self.meanmodes
        if self.calibration is not None:
            preds = (preds - self.meanmodes) * self.calibration + self.meanmodes
        if baseline is not None:
            preds = preds + baseline

        error = preds - modes
        self._accumulate(error, modes, reference)

        self.outputs['prediction'].value = preds
        self.outputs['prediction'].generation_time = self.current_time
        self.outputs['loss'].set_value(float(self.xp.mean(error ** 2)))
        self.outputs['loss'].generation_time = self.current_time

    def _accumulate(self, error, modes, reference):
        s = self.sums
        s['e'] += error.sum(axis=0)
        s['abs_e'] += self.xp.abs(error).sum(axis=0)
        s['e2'] += (error ** 2).sum(axis=0)
        # Symmetric mean absolute percentage error
        s['smape'] += (2.0 * self.xp.abs(error) / (self.xp.abs(modes) + self.xp.abs(error + modes) + 1e-3)).sum(axis=0)
        s['t'] += modes.sum(axis=0)
        s['t2'] += (modes ** 2).sum(axis=0)
        self.count += error.shape[0]
        if reference is not None:
            s['ref_e2'] += ((reference - modes) ** 2).sum(axis=0)
            self.reference_count += error.shape[0]

    def finalize(self):
        if self.count == 0:
            self.logger.warning('No data processed!')
            return super().finalize()

        n = self.count
        s = {k: cpuArray(v) for k, v in self.sums.items()}
        mae = s['abs_e'] / n
        mse = s['e2'] / n
        rmse = np.sqrt(mse)
        std_error = np.sqrt(np.maximum(mse - (s['e'] / n) ** 2, 0))
        smape = 100.0 * s['smape'] / n

        lines = [f'Test statistics over {n} samples', '',
                 f"{'Mode':<6} {'MAE':<12} {'RMSE':<12} {'SMAPE (%)':<12} {'StdErr':<12}"]
        lines += [f'{i:<6d} {mae[i]:<12.6f} {rmse[i]:<12.6f} {smape[i]:<12.2f} {std_error[i]:<12.6f}'
                  for i in range(self.nmodes)]
        lines += ['',
                  f'Mean absolute error (averaged): {np.mean(mae):.6f}',
                  f'Root mean squared error (averaged): {np.mean(rmse):.6f}',
                  f'Mean SMAPE: {np.mean(smape):.2f}%',
                  f'Maximum MAE across modes: {np.max(mae):.6f} (mode {np.argmax(mae)})',
                  f'Minimum MAE across modes: {np.min(mae):.6f} (mode {np.argmin(mae)})',
                  '']
        lines += self._group_table(s)
        self.logger.info('\n'.join(lines))

        self.mean_absolute_error = mae
        self.root_mean_squared_error = rmse
        self.smape = smape
        return super().finalize()

    def _group_table(self, s):
        """Per mode group: how much of the residual each estimator explains,
        on exactly the same frames. FVU = 1 means no better than predicting
        the mean; the lower of the two columns is the better estimator."""
        n = self.count
        var_mode = s['t2'] / n - (s['t'] / n) ** 2
        mse = s['e2'] / n
        ref_mse = s['ref_e2'] / self.reference_count if self.reference_count else None
        header = f"{'modes':>9} {'label_rms':>10} {'cnn_err':>9} {'cnn_FVU':>8}"
        if ref_mse is not None:
            header += f" | {'ref_err':>9} {'ref_FVU':>8}"
        lines = ['Per mode group (RMS in quadrature over the group, in the labels units):', header]
        for a, b in mode_groups(self.nmodes):
            v = max(float(var_mode[a:b].sum()), 1e-30)
            err = float(mse[a:b].sum())
            line = f"{f'{a}-{b - 1}':>9} {np.sqrt(v):>10.2f} {np.sqrt(err):>9.2f} {err / v:>8.3f}"
            if ref_mse is not None:
                ref = float(ref_mse[a:b].sum())
                line += f" | {np.sqrt(ref):>9.2f} {ref / v:>8.3f}"
            lines.append(line)
        total = f"{'total':>9} {np.sqrt(max(float(var_mode.sum()), 1e-30)):>10.2f} {np.sqrt(float(mse.sum())):>9.2f}"
        if ref_mse is not None:
            total += f" {'':>8} | {np.sqrt(float(ref_mse.sum())):>9.2f}"
        lines.append(total)
        return lines
