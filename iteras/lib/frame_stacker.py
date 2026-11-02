class FrameStacker:
    """
    Stacks each frame with the n_frames - 1 frames before it along the
    channel axis: sample t becomes [x(t), x(t-1), ..., x(t-n_frames+1)].

    Frames arrive in batches, rows in time order (as buffered by a
    DataBuffer), or one at a time (a batch of 1); the last n_frames - 1 frames
    are carried over to the next call. At the very start, when there is no
    history yet, the missing frames are filled by repeating the first one.

    Works on numpy or cupy arrays (xp), of shape (B, C, H, W), or (B, H, W)
    for single-channel frames, which are treated as C = 1. The result is
    always (B, C * n_frames, H, W).

    Stacking must happen on the complete, contiguous frame sequence, before
    any frames are dropped (e.g. by gain_mod filtering): each kept sample
    then still carries its true preceding frames.
    """

    def __init__(self, n_frames, xp):
        if n_frames < 1:
            raise ValueError(f'n_frames must be >= 1, got {n_frames}')
        self.n_frames = n_frames
        self.xp = xp
        self._previous = None

    def __call__(self, x):
        if self.n_frames == 1:
            return x
        xp = self.xp
        if x.ndim == 3:
            x = x[:, xp.newaxis]
        n = self.n_frames
        previous = self._previous
        if previous is None:
            previous = xp.repeat(x[:1], n - 1, axis=0)
        sequence = xp.concatenate([previous, x], axis=0)
        batch = x.shape[0]
        stacked = xp.concatenate([sequence[n - 1 - k:n - 1 - k + batch] for k in range(n)], axis=1)
        self._previous = sequence[-(n - 1):].copy()
        return stacked
