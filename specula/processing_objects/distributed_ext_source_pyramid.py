import copy

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.ext_source_pyramid import ExtSourcePyramid
import specula


class DistributedExtSourcePyramid(ExtSourcePyramid):
    """
    ExtSourcePyramid that distributes work on multiple devices.

    Internally, it manages a series of hidden ExtSourcePyramid objects, each
    computing the focal plane PSF and the pyramid image of a slice of the
    extended source points (see the n_point_slices parameter of ExtSourcePyramid).
    In post_trigger(), the partial images are summed and the outputs are
    computed as in ExtSourcePyramid.

    The parameters are those of ExtSourcePyramid, plus n_slices: the number
    of slices of the source points, each one on the next device (modulo the
    number of devices).
    """
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 fov: float,
                 pup_diam: float,
                 output_resolution: int,
                 n_slices: int,
                 fov_errinf: float = 0.5,
                 fov_errsup: float = 2,
                 pup_dist: float = None,
                 pup_margin: int = 2,
                 fft_res: float = 3.0,
                 fp_obs: float = None,
                 pup_shifts = (0.0, 0.0),
                 pyr_tlt_coeff: float = None,
                 pyr_edge_def_ld: float = 0.0,
                 pyr_tip_def_ld: float = 0.0,
                 pyr_tip_maya_ld: float = 0.0,
                 pyr_max_side_ld: float = 0.0,
                 min_pup_dist: float = None,
                 rotAnglePhInDeg: float = 0.0,
                 xShiftPhInPixel: float = 0.0,
                 yShiftPhInPixel: float = 0.0,
                 max_batch_size: int = 128,
                 max_flux_ratio_thr: float = 1e-3,
                 cuda_stream_enable: bool = True,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        # Complete dict of init arguments, without extra ones
        args = copy.copy(locals())
        del args['self']
        del args['__class__']
        del args['n_slices']

        # Initialize base class - we do not use the calculation routines,
        # but it is needed for inputs, outputs and the final post-processing
        super().__init__(**args)

        # Initialize the internal pyramids, one for each slice of the source points.
        # If using GPUs, each one targets a different device.
        self.sub_pyr = []
        for i in range(n_slices):
            # The device of this object (target_device_idx may be None: the default one)
            if self.target_device_idx >= 0:  # pragma: gpu
                num_devices = specula.cp.cuda.runtime.getDeviceCount()
                args['target_device_idx'] = (self.target_device_idx + i) % num_devices
            args['n_point_slices'] = n_slices
            args['point_slice_idx'] = i
            self.sub_pyr.append(ExtSourcePyramid(**args))
        # Building the slices made their devices current: back to the one of this object
        if self.target_device_idx >= 0:  # pragma: gpu
            self._target_device.use()

        # Buffer for the partial images of the slices, copied to this device
        self._slice_image = self.xp.zeros_like(self.fpsf)

    @classmethod
    def devices_used(cls, params):
        return int(params.get('n_slices', 1))

    @classmethod
    def single_device_params(cls, params):
        return {**params, 'n_slices': 1}

    @classmethod
    def input_names(cls):
        return super().input_names()

    @classmethod
    def output_names(cls):
        return super().output_names()

    def setup(self):
        '''
        Skip the ExtSourcePyramid method for this object, since we do not
        compute the slices, but call the BaseProcessingObj one for housekeeping
        and prepare the final post-processing. Then set inputs on all sub-pyramids
        '''
        BaseProcessingObj.setup(self)
        self._make_pup_shift_interp()
        self._update_s0_area()

        # Copy our inputs into all sub-pyramids
        for i, pyr in enumerate(self.sub_pyr):
            pyr.name = f'subpyr{i}'
            for k, v in self.inputs.items():
                if len(v.input_values) > 0:
                    pyr.inputs[k].set(v.input_values[0].cloned_value)
            pyr.setup()
        if self.target_device_idx >= 0:  # pragma: gpu
            self._target_device.use()

    def check_ready(self, t):
        '''
        Call the BaseProcessingObj method for housekeeping, then all sub-pyramids
        '''
        BaseProcessingObj.check_ready(self, t)
        for pyr in self.sub_pyr:
            pyr.check_ready(t)

    def prepare_trigger(self, t):
        '''
        Only the BaseProcessingObj method for housekeeping. As in DistributedSH,
        the sub-pyramids prepare in their own check_ready(), after getting their inputs.
        '''
        BaseProcessingObj.prepare_trigger(self, t)

    def trigger(self):
        '''
        Call the BaseProcessingObj method for housekeeping, then all sub-pyramids
        '''
        BaseProcessingObj.trigger(self)
        for pyr in self.sub_pyr:
            pyr.trigger()

    def trigger_code(self):
        '''
        Nothing to do here. The sub-pyramids run the regular ExtSourcePyramid
        implementation.
        '''
        return

    def post_trigger(self):
        '''
        Sum the partial focal plane PSFs and pyramid images of the sub-pyramids,
        then compute the outputs with the ExtSourcePyramid method.
        '''
        # The sub-pyramids made their devices current
        if self.target_device_idx >= 0:  # pragma: gpu
            self._target_device.use()
        self.fpsf[:] = 0
        self.pyr_image[:] = 0
        for pyr in self.sub_pyr:
            # Housekeeping, and wait for the CUDA graph of the sub-pyramid, if any
            BaseProcessingObj.post_trigger(pyr)
            if pyr.target_device_idx >= 0 and pyr.target_device_idx != self.target_device_idx:  # pragma: gpu
                # The copy below is not ordered with the work of another device
                pyr._target_device.synchronize()
                self._target_device.use()
            self._slice_image[:] = pyr.fpsf
            self.fpsf += self._slice_image
            self._slice_image[:] = pyr.pyr_image
            self.pyr_image += self._slice_image

        # The sub-pyramids apply in_misregistration: follow their pupil shifts
        pup_shifts = tuple(float(v) for v in self.sub_pyr[0].pup_shifts)
        if pup_shifts != tuple(float(v) for v in self.pup_shifts):
            self.pup_shifts = pup_shifts
            self._make_pup_shift_interp()
        self._update_s0_area()

        # Normalization of the PSFs to the total flux, as in ExtSourcePyramid.cache_ttexp()
        # (the flux filtering moves the flux of the filtered points, keeping the total)
        coeff = self.local_inputs['ext_source_coeff'].value
        self.factor = 1.0 / (self.xp.sum(coeff[:, 3]) + 1e-20)

        super().post_trigger()
