import numpy as np

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels

# 1 nm RMS of tip/tilt mode amplitude = 4e-9/38.5 rad on sky, at 7.5 mas/px (MORFEO LO sensor)
NM_TO_PX = (4e-9 / 38.5 * 3600 * 180 / np.pi * 1e3) / 7.5


class SimpleSpotSupervisor(BaseProcessingObj):
    """
    Spot reacquisition for a single-subaperture, fixed-window WCoG tip/tilt sensor (e.g. the MORFEO LO NGS sensor).

    The slopes are computed by a plain :class:`ShSlopec` whose window never moves. This object only watches the
    detector frames and, when the spot is detected outside the capture radius of the window, issues a tip/tilt
    feedforward that brings the spot back to the reference.

    Every ``look_frames`` detector frames (one *look*), the mean frame is background-subtracted (median) and
    correlated (FFT) with a Gaussian template of FWHM ``template_fwhm``: this is the least-squares fit of a Gaussian
    of fixed width at every position of the field. The correlation is normalised to a signal-to-noise ratio with a
    robust (MAD) estimate of the pixel noise of the look itself, so that the threshold is a plain number of sigmas,
    independent of flux, background and frame time. If the global SNR peak is above ``snr_thr``, lies farther than
    ``capture_radius`` from the reference, and the SNR within ``capture_radius`` of the reference is below
    ``snr_thr`` (the window does not see the spot), the peak position is added to the feedforward. The next
    ``cmd_latency`` frames are discarded, so that no look mixes frames taken before and after the feedforward
    reached the pixels.

    There is no registration with the loop command, no consensus, no presence/dropout logic, no hold, and no window
    move: while the spot is outside the window the WCoG slope is about zero and the loop is effectively held.

    Parameters
    ----------
    look_frames : int
        Detector frames averaged in one look.
    snr_thr : float
        Detection threshold on the peak SNR of a look (sigmas).
    cmd_latency : int
        Frames from the feedforward output of this object to its effect on the pixels.
    capture_radius : float [pixels], optional
        Spots closer than this to the reference are left to the WCoG. Default 4.
    template_fwhm : float [pixels], optional
        FWHM of the Gaussian template. Default 3.
    np_sub : int, optional
        Side of the (square) detector frame in pixels. Default 240.
    cmd_to_px : list, optional
        2x2 matrix from tip/tilt command [nm] to the spot shift it compensates [px] (x = column, y = row).
        Default: the MORFEO LO sensor scale on the diagonal.

    Outputs: ``out_feedforward`` = accumulated feedforward tip/tilt command [nm] (feed it to an extra tip/tilt DM
    through the same delay as the loop command), ``out_state`` = telemetry [est_x, est_y, snr, snr_local, n_moves]
    of the last look.
    """

    def __init__(self,
                 look_frames: int,
                 snr_thr: float,
                 cmd_latency: int,
                 capture_radius: float = 4.0,
                 template_fwhm: float = 3.0,
                 np_sub: int = 240,
                 cmd_to_px: list = None,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if int(look_frames) < 1:
            raise ValueError('look_frames must be >= 1')
        if int(cmd_latency) < 0:
            raise ValueError('cmd_latency must be >= 0')
        self.n = int(np_sub)
        self.look_frames = int(look_frames)
        self.snr_thr = float(snr_thr)
        self.cmd_latency = int(cmd_latency)
        self.capture_radius = float(capture_radius)

        cmd = np.asarray(cmd_to_px if cmd_to_px is not None else np.eye(2) * NM_TO_PX, dtype=float)
        self.px_to_cmd = np.linalg.inv(cmd.reshape(2, 2))

        # Gaussian template with unit peak, centred on index 0 (wrapped), so that the correlation
        # peak index of a spot is its pixel index
        idx = np.arange(self.n)
        wrapped = np.minimum(idx, self.n - idx)
        sigma = template_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        tpl = np.exp(-(wrapped[:, None] ** 2 + wrapped[None, :] ** 2) / (2.0 * sigma ** 2))
        # correlation / (pixel noise * tpl_norm) = SNR of the fitted Gaussian amplitude
        self.tpl_norm = float(np.sqrt(np.sum(tpl ** 2)))
        self.tpl_conj = self.to_xp(np.conj(np.fft.fft2(tpl)), dtype=self.complex_dtype)

        self.c = (self.n - 1) / 2.0
        coord = idx - self.c
        local = (coord[:, None] ** 2 + coord[None, :] ** 2) <= self.capture_radius ** 2
        self.local_idx = self.to_xp(np.flatnonzero(local))

        self.acc = self.xp.zeros((self.n, self.n), dtype=self.dtype)
        self.count = 0
        self.skip = 0
        self.ff_px = np.zeros(2)
        self.n_moves = 0
        self.state = np.full(5, np.nan)
        self.state[4] = 0

        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.feedforward = BaseValue(value=self.xp.zeros(2, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.state_out = BaseValue(value=self.xp.zeros(5, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.outputs['out_feedforward'] = self.feedforward
        self.outputs['out_state'] = self.state_out

    @classmethod
    def input_names(cls):
        return {'in_pixels': InputDesc(Pixels, 'Detector pixels')}

    @classmethod
    def output_names(cls):
        return {'out_feedforward': OutputDesc(BaseValue, 'Accumulated feedforward tip/tilt command [nm]'),
                'out_state': OutputDesc(BaseValue, 'Telemetry of the last look [est_x, est_y, snr, snr_local, '
                                                   'n_moves]')}

    def look(self, image):
        """Locate the spot in a look image: returns (position [px from the centre], peak SNR, SNR near the
        reference)."""
        xp = self.xp
        med = xp.median(image)
        noise = 1.4826 * xp.median(xp.abs(image - med))
        corr = xp.real(xp.fft.ifft2(xp.fft.fft2(image - med) * self.tpl_conj))
        snr = corr / (noise * self.tpl_norm)
        i = xp.argmax(snr)
        res = cpuArray(xp.stack([i.astype(self.dtype), snr.ravel()[i], xp.max(snr.ravel()[self.local_idx])]))
        row, col = divmod(int(res[0]), self.n)
        return np.array([col - self.c, row - self.c]), float(res[1]), float(res[2])

    def process_frame(self, frame):
        """One detector frame; returns True if a feedforward was issued."""
        if self.skip > 0:
            self.skip -= 1
            return False
        self.acc += frame
        self.count += 1
        if self.count < self.look_frames:
            return False
        pos, snr, snr_local = self.look(self.acc / self.look_frames)
        self.acc[:] = 0
        self.count = 0
        moved = snr > self.snr_thr and np.hypot(*pos) > self.capture_radius and snr_local < self.snr_thr
        if moved:
            self.ff_px = self.ff_px + pos
            self.n_moves += 1
            self.skip = self.cmd_latency
        self.state[:] = [pos[0], pos[1], snr, snr_local, self.n_moves]
        return moved

    def trigger_code(self):
        self.process_frame(self.local_inputs['in_pixels'].pixels)
        self.feedforward.value[:] = self.to_xp(self.px_to_cmd @ self.ff_px, dtype=self.dtype)
        self.state_out.value[:] = self.to_xp(self.state, dtype=self.dtype)

    def post_trigger(self):
        super().post_trigger()
        self.feedforward.generation_time = self.current_time
        self.state_out.generation_time = self.current_time
