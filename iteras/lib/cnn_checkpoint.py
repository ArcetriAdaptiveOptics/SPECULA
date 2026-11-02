"""
Saving and loading the networks trained by Conv2dNetTrainer.

A checkpoint is two files:

- ``<name>.pth``: the network weights (a state_dict);
- ``<name>_stats.json``: everything else needed to use them -- the network's
  architecture (``network``, the arguments it was built with, see
  NETWORK_KEYS), the input and output normalization (meanp/stdp,
  meanmodes/stdmodes) and the per-mode prediction gain (``mode_gain``, see
  calibration_factor).

Code that uses a trained network reads all of this from the file, so its own
configuration cannot disagree with the weights.
"""

import json
import os

import numpy as np
import torch

from iteras.lib.efficient_u_net import UNetRegressor


# The arguments a network is built with, as saved in the stats file, and the
# defaults of the ones added later (absent from older stats files).
NETWORK_KEYS = ('nmodes', 'input_channels', 'n_frames', 'channels', 'depth',
                'conv_block_type', 'head_type', 'head_grid', 'dropout', 'channel_product')
NETWORK_DEFAULTS = {'channel_product': False}

# A network's predictions are shrunk towards the mean, by a factor that differs
# per mode (see Conv2dNetTrainer's mode_gain). Dividing it out restores the
# amplitude, but on modes the network barely predicts that would mostly amplify
# noise, so the correction is bounded on both ends.
MIN_CALIBRATED_GAIN = 0.2
MAX_CALIBRATION = 3.0


def build_network(network):
    """A UNetRegressor from a ``network`` dict (keys: NETWORK_KEYS)."""
    return UNetRegressor(
        input_channels=network['input_channels'] * network['n_frames'],
        output_size=network['nmodes'],
        base_channels=network['channels'],
        dropout_level=network['dropout'],
        depth=network['depth'],
        conv_block_type=network['conv_block_type'],
        head_type=network['head_type'],
        head_grid=network['head_grid'],
    )


def network_input(x, network):
    """The network input, one frame per row, from a batch of maps x: (B, H, W)
    for one map per frame, or (B, C, H, W) with C channels. With the network's
    channel_product the input is the per-pixel product of the two channels of
    x (e.g. amplitude and phase), one map per frame; otherwise x itself,
    whose channels must match the network's input_channels."""
    if network['channel_product']:
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f'channel_product needs maps with 2 channels, got shape {x.shape}')
        return x[:, 1] * x[:, 0]
    channels = 1 if x.ndim == 3 else x.shape[1]
    if channels != network['input_channels']:
        raise ValueError(f"the input maps have {channels} channel(s) (shape {x.shape}), but the network "
                         f"has input_channels={network['input_channels']}. Set input_channels to "
                         f"{channels}, or channel_product to use the product of 2 channels")
    return x


def predict(model, inputs, mean, std, chunk=128):
    """Predictions of ``model`` in physical units (denormalized with the
    per-mode mean/std tensors), in eval mode and without gradients, in
    chunks of at most ``chunk`` samples so that memory doesn't grow with the
    number of samples. The default keeps the peak low on a GPU shared with
    the simulation: a U-Net with 64 channels on 120x120 maps needs ~30 GiB
    for 1000 samples, ~4 GiB for 128, and torch's caching allocator keeps
    that memory reserved afterwards."""
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)])
    return out * std + mean


def calibration_factor(stats, nmodes):
    """Per-mode factor that undoes the shrinkage measured during training, to
    multiply the predictions' deviation from meanmodes by. None if the
    checkpoint has no measurement."""
    gain = np.asarray(stats.get('mode_gain') or [], dtype=float)
    if gain.size != nmodes:
        return None
    factor = np.ones(nmodes)
    correctable = gain > MIN_CALIBRATED_GAIN
    factor[correctable] = np.minimum(1.0 / gain[correctable], MAX_CALIBRATION)
    return factor


def stats_filename(network_filename):
    return os.path.splitext(network_filename)[0] + '_stats.json'


def load_stats(network_filename):
    filename = stats_filename(network_filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'Statistics file not found at {filename}. '
                                f'Make sure to train the model first!')
    with open(filename, 'r') as f:
        return json.load(f)


def _saved_network(network_filename, stats):
    network = stats.get('network')
    if network is None:
        raise ValueError(f'{stats_filename(network_filename)} does not record the network '
                         f'architecture: the checkpoint predates that, retrain it')
    return {**NETWORK_DEFAULTS, **network}


def check_same_network(network_filename, network):
    """Raise a clear error if the checkpoint at network_filename was trained
    with a different network than ``network`` -- for a trainer resuming from
    it. Does nothing if there is no checkpoint yet."""
    if not os.path.isfile(stats_filename(network_filename)):
        return
    trained = _saved_network(network_filename, load_stats(network_filename))
    diffs = [f'{k}={network[k]!r} (checkpoint: {trained.get(k)!r})'
             for k in NETWORK_KEYS if network[k] != trained[k]]
    if diffs:
        raise ValueError(f'the network configured does not match the one in {network_filename}: '
                         + ', '.join(diffs) + '. Use the checkpoint\'s values, or a new network_filename '
                         'to train a different network')


def load_weights(model, network_filename):
    model.load_state_dict(torch.load(network_filename, map_location='cpu', weights_only=True))


def save_checkpoint(model, network_filename, stats):
    os.makedirs(os.path.dirname(network_filename) or '.', exist_ok=True)
    torch.save(model.state_dict(), network_filename)
    with open(stats_filename(network_filename), 'w') as f:
        json.dump(stats, f, indent=2)


def load_trained_network(network_filename):
    """Build the network recorded in the checkpoint, load its weights and
    return it in eval mode, on the CPU, together with its stats (a dict; the
    architecture is in stats['network'])."""
    if not os.path.isfile(network_filename):
        raise FileNotFoundError(f'Model file not found at {network_filename}')
    stats = load_stats(network_filename)
    stats['network'] = _saved_network(network_filename, stats)
    model = build_network(stats['network'])
    load_weights(model, network_filename)
    return model.eval(), stats
