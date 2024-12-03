class GaussianConvolutionKernel(ConvolutionKernel):
    """
    Kernel processing object for Gaussian kernels.
    """

    def __init__(self, spotsize=None):
        super().__init__('kernel_gauss', 'kernel_gauss processing object')
        self._kernels = BaseValue()
        self._spotsize = spotsize
        self._last_recalc = None

    def set_property(self, spotsize=None, **kwargs):
        """
        Sets the properties for the Gaussian kernel.
        """
        super().set_property(**kwargs)
        if spotsize is not None:
            self.set_spotsize(spotsize)

    def get_property(self, spotsize=None, **kwargs):
        """
        Gets the properties of the Gaussian kernel.
        """
        super().get_property(**kwargs)
        if 'spotsize' in kwargs:
            kwargs['spotsize'] = self._spotsize

    def set_spotsize(self, spotsize):
        if spotsize != self._spotsize:
            self._spotsize = spotsize
            self._recalc = True

    def trigger(self, t):
        """
        Trigger recalculation if needed and update the kernel's generation time.
        """
        if self._recalc:
            self.recalc()
            self._recalc = False
            self._last_recalc = t

        self._kernels.generation_time = t

    def recalc(self):
        """
        Recalculates the Gaussian kernel based on current settings.
        """
        dimx = max(self._lenslet.dimx, 2)
        pupil_size_m = self._ef.pixel_pitch * self._ef.size[0]

        print("Calculating gaussian kernel...")
        lgs_tt = [-0.5, -0.5] if not self._positiveShiftTT else [0.5, 0.5]
        lgs_tt = [x * self._pxscale for x in lgs_tt]

        if obj_valid(self._cm):
            arr = [
                self._lenslet.dimx, pupil_size_m, 90e3, self._spotsize,
                self._pxscale, self._dimension, 3, lgs_tt, [0, 0, 0], [90e3], [1.0]
            ]
            sha1 = jc_sha1(json_serialize(arr))
            kk = self._cm.read_kernel(sha1)
        else:
            kk = None

        if kk is None:
            print("Calculating kernel...")
            kk = lgs_map_sh(
                dimx, pupil_size_m, 0, 90e3, [0], oversampling=1, zprofile=[1.0],
                spotsize=self._spotsize, pxscale=self._pxscale, dimension=self._dimension,
                lgs_tt=lgs_tt, cube=True, progress=True, precision=self.precision
            )
            if obj_valid(self._cm):
                self._cm.write_kernel(sha1, kk)

        if self._lenslet.dimx != dimx:
            kk = kk[:, :, 0]

        for i in range(self._lenslet.dimx):
            for j in range(self._lenslet.dimy):
                subap_kern = np.array(kk[:, :, j * self._lenslet.dimx + i])
                subap_kern /= np.sum(subap_kern)

                if self._returnFft:
                    subap_kern_fft = np.fft.ifftshift(np.fft.ifft2(subap_kern))
                    self._kernels.ptr_value[j * self._lenslet.dimx + i] = subap_kern_fft
                else:
                    self._kernels.ptr_value[j * self._lenslet.dimx + i] = subap_kern

    def revision_track(self):
        """
        Returns the revision of the SVN.
        """
        return "$Rev$"

    def run_check(self, time_step, errmsg=""):
        """
        Run a check before the simulation to ensure that all necessary inputs have been set.
        Returns True if everything is set up correctly, False otherwise.
        """
        valid = obj_valid(self._lenslet) and obj_valid(self._ef)
        if not obj_valid(self._lenslet):
            errmsg += "Kernel (kernel_gauss) lenslet object is not valid"
        if not obj_valid(self._ef):
            errmsg += "Kernel (kernel_gauss) ef object is not valid"

        return valid
