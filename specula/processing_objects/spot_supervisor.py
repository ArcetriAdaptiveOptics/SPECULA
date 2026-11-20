import numpy as np

from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels

# 1 nm RMS of tip/tilt mode amplitude = 4e-9/38.5 rad on sky, at 7.5 mas/px (MORFEO LO sensor)
NM_TO_PX = (4e-9 / 38.5 * 3600 * 180 / np.pi * 1e3) / 7.5


class SpotSupervisor(BaseProcessingObj):
    """
    Spot (re)acquisition for a single-subaperture WCoG tip/tilt sensor (e.g. the MORFEO LO NGS sensor), with one
    image and one statistic for every decision.

    Every ``look_frames`` detector frames (one look) the mean frame minus its median is correlated (FFT) with a
    unit-peak Gaussian of FWHM ``template_fwhm`` and normalised to a signal-to-noise ratio with the pixel noise of
    the look (1.4826 * MAD) times the template norm. From this SNR map: p, s = global peak and its SNR; s_loc =
    highest 3x3 maximum within ``capture_radius`` of the window w. Per look:

    - hold: the window is empty if no SNR value within ``capture_radius`` of it exceeds ``presence_thr``. After
      ``k_absent`` empty looks the loop is held (:class:`ShSlopecMovable` outputs zero slopes; with ``out_hold`` on
      the controller's ``in_freeze`` the command stops). A star dropout and a spot carried away by a step are the
      same case: the WCoG has nothing to measure. Moves stay possible during the hold; the first look with a spot
      in the window ends it.
    - move: if s > ``snr_thr``, |p - w| > ``capture_radius``, s_loc < ``q_thr`` * s and no window return is in
      flight, the window (slope zero of ShSlopecMovable) is put on p: the loop stabilises the spot where it is.
      s_loc counts peaks only: a spot just outside the capture radius puts its own wing (not a peak) in the disk and
      must not veto its own recovery; a spot in the window vetoes moves to peaks weaker than s_loc / q_thr.
    - confirmation: the first look after a move confirms it if the SNR map has a value > ``presence_thr`` within
      ``capture_radius`` of the new window; then a feedforward tip/tilt equal to the window offset is issued and the
      window returns to the reference when the feedforward reaches the pixels (``cmd_latency`` frames). Otherwise
      the move is aborted and the window returns to the reference (moves start only from the reference: no move
      while a move is being confirmed or a return is in flight).

    The look restarts at every window change. Decisions come once per look: speed is traded for robustness.

    Parameters
    ----------
    cmd_latency : int
        Frames from the feedforward output of this object to its effect on the pixels (>= 1).
    look_frames : int
        Detector frames averaged in one look.
    capture_radius : float [pixels]
        Spots closer than this to the window are left to the WCoG.
    k_absent : int
        Consecutive looks with an empty window that start the hold.
    snr_thr : float, optional
        SNR of a look needed to move the window [sigma]. Default 7.5.
    q_thr : float, optional
        Local veto: no move if s_loc >= q_thr * s. Default 0.7.
    presence_thr : float, optional
        SNR within the capture radius needed to count the window as occupied, and to confirm a move [sigma].
        Default 5.5: the noise-only maximum within a 4.2-px disk of 48-ms looks with the MORFEO LO EMCCD noise
        exceeds 5.0 in 4e-4 of the looks and 5.5 in none of 3e4 (the global maximum, 52x more pixels, exceeds 6.5
        in 1e-3).
    template_fwhm : float [pixels], optional
        FWHM of the Gaussian template. Default 3.
    np_sub : int, optional
        Side of the (square) detector frame [pixels]. Default 240.
    cmd_to_px : list, optional
        2x2 matrix from tip/tilt command [nm] to the spot shift it compensates [px] (x = column, y = row).
        Default: the MORFEO LO sensor scale on the diagonal.

    Outputs: ``out_window`` = [wx, wy, hold] (feed it to ShSlopecMovable, delayed by one step),
    ``out_feedforward`` = accumulated feedforward command [nm] (feed it to an extra tip/tilt DM),
    ``out_hold`` = the hold flag alone (feed it to the controller's ``in_freeze``, delayed by one step, so that
    the command does not keep moving with the controller memory while the loop is held),
    ``out_state`` = telemetry of the last look [est_x, est_y, s, s_loc, dropout, n_moves, n_aborts],
    ``out_look`` = mean frame of the last look. Outputs are refreshed only in steps with new pixels.
    """

    def __init__(self,
                 cmd_latency: int,
                 look_frames: int,
                 capture_radius: float,
                 k_absent: int,
                 snr_thr: float = 7.5,
                 q_thr: float = 0.7,
                 presence_thr: float = 5.5,
                 template_fwhm: float = 3.0,
                 np_sub: int = 240,
                 cmd_to_px: list = None,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if int(cmd_latency) < 1 or int(look_frames) < 1 or int(k_absent) < 1:
            raise ValueError('cmd_latency, look_frames and k_absent must be >= 1')
        self.cmd_latency = int(cmd_latency)
        self.look_frames = int(look_frames)
        self.capture_radius = float(capture_radius)
        self.k_absent = int(k_absent)
        self.snr_thr = float(snr_thr)
        self.q_thr = float(q_thr)
        self.presence_thr = float(presence_thr)
        self.n = int(np_sub)

        cmd = np.asarray(cmd_to_px if cmd_to_px is not None else np.eye(2) * NM_TO_PX, dtype=float)
        self.px_to_cmd = np.linalg.inv(cmd.reshape(2, 2))

        # unit-peak Gaussian centred on index 0 (wrapped): the correlation peak index of a spot is its pixel index,
        # and correlation / (pixel noise * ||G||) is the SNR of a Gaussian fit of fixed width at each pixel
        idx = np.arange(self.n)
        wrapped = np.minimum(idx, self.n - idx)
        sigma = template_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        g = np.exp(-(wrapped[:, None] ** 2 + wrapped[None, :] ** 2) / (2.0 * sigma ** 2))
        self.g_norm = float(np.sqrt(np.sum(g ** 2)))
        self.g_conj = self.to_xp(np.conj(np.fft.fft2(g)), dtype=self.complex_dtype)
        self.c = (self.n - 1) / 2.0
        self.coord = idx - self.c                                 # pixel index -> px from the centre
        self._disk = None                                         # (window, disk flat indices, 8 neighbours)

        self.w = np.zeros(2)                                      # window centre [px]
        self.u_ff = np.zeros(2)                                   # accumulated feedforward [px]
        self.frame = 0
        self.look_acc, self.look_count = None, 0
        self.last_look = self.xp.zeros((self.n, self.n), dtype=self.dtype)
        self.miss = 0
        self.dropout = False
        self.to_confirm = False
        self.w_queue = []                                         # [(frame, window)] returns waiting for the ff
        self.n_moves = 0
        self.n_aborts = 0
        self.state = [np.nan, np.nan, np.nan, np.nan]             # est_x, est_y, s, s_loc of the last look
        self._processed = False

        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.window = BaseValue(value=self.xp.zeros(3, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.feedforward = BaseValue(value=self.xp.zeros(2, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.state_out = BaseValue(value=self.xp.zeros(7, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.look_out = BaseValue(value=self.xp.zeros((self.n, self.n), dtype=self.dtype),
                                  target_device_idx=self.target_device_idx)
        self.outputs['out_window'] = self.window
        self.outputs['out_feedforward'] = self.feedforward
        self.outputs['out_state'] = self.state_out
        self.outputs['out_look'] = self.look_out
        self.hold_out = BaseValue(value=self.xp.zeros(1, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self.outputs['out_hold'] = self.hold_out

    @classmethod
    def input_names(cls):
        return {'in_pixels': InputDesc(Pixels, 'Detector pixels')}

    @classmethod
    def output_names(cls):
        return {'out_window': OutputDesc(BaseValue, 'Window centre [wx, wy] px and hold flag'),
                'out_feedforward': OutputDesc(BaseValue, 'Accumulated feedforward tip/tilt command [nm]'),
                'out_hold': OutputDesc(BaseValue, 'Hold flag of out_window (e.g. for IirFilter.in_freeze)'),
                'out_look': OutputDesc(BaseValue, 'Mean frame of the last look [e-]'),
                'out_state': OutputDesc(BaseValue, 'Telemetry of the last look [est_x, est_y, s, s_loc, dropout, '
                                                   'n_moves, n_aborts]')}

    def _window_disk(self):
        """Flat indices of the pixels within capture_radius of the window and (8, n_disk) flat indices of their
        8 neighbours (wrapped); cached per window."""
        key = (float(self.w[0]), float(self.w[1]))
        if self._disk is None or self._disk[0] != key:
            d2 = (self.coord[None, :] - key[0]) ** 2 + (self.coord[:, None] - key[1]) ** 2
            flat = np.flatnonzero(d2 <= self.capture_radius ** 2)
            row, col = np.divmod(flat, self.n)
            nb = np.stack([((row + dy) % self.n) * self.n + (col + dx) % self.n
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
            self._disk = (key, self.to_xp(flat), self.to_xp(nb))
        return self._disk[1], self._disk[2]

    def _snr(self, image):
        """p [px from the centre], s, s_max_in_disk, s_loc (highest 3x3 maximum in the disk) of one look."""
        xp = self.xp
        med = xp.median(image)
        noise = 1.4826 * xp.median(xp.abs(image - med))
        snr = (xp.real(xp.fft.ifft2(xp.fft.fft2(image - med) * self.g_conj)) / (noise * self.g_norm)).ravel()
        i = int(xp.argmax(snr))
        disk, nb = self._window_disk()
        vals = snr[disk]
        s_disk = float(xp.max(vals))
        s_loc = float(xp.max(xp.where(vals >= xp.max(snr[nb], axis=0), vals, -xp.inf)))
        row, col = divmod(i, self.n)
        return np.array([col - self.c, row - self.c]), float(snr[i]), s_disk, s_loc

    def _set_window(self, w_new):
        self.w = np.asarray(w_new, dtype=float)
        self.look_count = 0                                       # the next look starts at the new window

    def _decide(self, image):
        """All decisions of one complete look."""
        p, s, s_disk, s_loc = self._snr(image)
        self.state = [p[0], p[1], s, s_loc]
        # hold: no spot in the window for k_absent looks (star absent, or spot carried away): the WCoG has nothing
        # to measure. Moves stay possible during the hold
        self.miss = 0 if s_disk > self.presence_thr else self.miss + 1
        self.dropout = self.miss >= self.k_absent
        # confirmation of the previous move: is there a spot at the new window?
        if self.to_confirm:
            self.to_confirm = False
            if s_disk > self.presence_thr:
                self.u_ff = self.u_ff + self.w
                # the window returns when the feedforward reaches the pixels: a window set at step k acts on frame
                # k+1, the feedforward issued at step k on frame k + cmd_latency
                self.w_queue.append((self.frame + self.cmd_latency - 1, np.zeros(2)))
            else:
                self.n_aborts += 1
                self._set_window(np.zeros(2))
            return
        # move
        far = np.linalg.norm(p - self.w) > self.capture_radius
        if s > self.snr_thr and far and s_loc < self.q_thr * s and not self.w_queue:
            self.n_moves += 1
            self._set_window(p)
            self.to_confirm = True

    def process_frame(self, frame):
        """One detector frame."""
        frame = self.xp.asarray(frame, dtype=self.dtype)
        self.look_acc = frame.copy() if self.look_count == 0 else self.look_acc + frame
        self.look_count += 1
        if self.look_count == self.look_frames:
            self.look_count = 0
            self.last_look = self.look_acc / self.look_frames
            self._decide(self.last_look)
        while self.w_queue and self.w_queue[0][0] <= self.frame:
            self._set_window(self.w_queue.pop(0)[1])
        self.frame += 1

    def trigger_code(self):
        # frames only: a step without new pixels (another input refreshed) is not processed
        self._processed = self.local_inputs['in_pixels'].generation_time >= self.current_time
        if not self._processed:
            return
        self.process_frame(self.local_inputs['in_pixels'].pixels)
        self.window.value[:] = self.to_xp([self.w[0], self.w[1], float(self.dropout)], dtype=self.dtype)
        self.hold_out.value[:] = self.window.value[2]
        self.feedforward.value[:] = self.to_xp(self.px_to_cmd @ self.u_ff, dtype=self.dtype)
        self.state_out.value[:] = self.to_xp([*self.state, float(self.dropout), self.n_moves, self.n_aborts],
                                             dtype=self.dtype)
        if self.look_count == 0:                                  # a look completed (or restarted) in this frame
            self.look_out.value[:] = self.last_look

    def post_trigger(self):
        super().post_trigger()
        if not self._processed:                                   # no new frame in this step: outputs not refreshed
            return
        for name in ('out_window', 'out_hold', 'out_feedforward', 'out_state', 'out_look'):
            self.outputs[name].generation_time = self.current_time
