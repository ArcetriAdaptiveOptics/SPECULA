from specula.processing_objects.slopec import Slopec
from specula.base_processing_obj import InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.pupdata import PupData
from specula.lib.make_mask import make_mask


class CurWfsSlopec(Slopec):
    """
    Slope Computer for Curvature Wavefront Sensor processing object.
    Computes the normalized difference between intra-focal and extra-focal fluxes
    on a pixel-by-pixel basis using PupData indices.
    """
    def __init__(self,
                 diameter: int,
                 ccd_size: tuple,
                 sn: Slopes = None,
                 interleave: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Parameters:
        ----------

        diameter: int
            Diameter of the pupil in pixels (used to define valid pupil indices).
        ccd_size: tuple
            Size of the CCD in pixels (height, width).
        sn: Slopes, optional
            Slopes object for reference subtraction (if needed).
        interleave: bool, optional
            Whether to interleave slopes (not used in this implementation).
        target_device_idx : int, optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        precision : int, optional
            Precision for computation (0 for double, 1 for single). Default is None
            (uses global setting).
        """

        cx = ccd_size[1]/2
        cy = ccd_size[0]/2

        _,ids = make_mask(np_size=ccd_size[0],
                          diaratio = diameter/float(ccd_size[0]),
                          get_idx=True)
        mask_ids = ids[0]*ccd_size[1]+ids[1]

        self.pupdata = PupData(
            ind_pup=mask_ids,
            radius=diameter/2,
            cx=cx, cy=cy,
            framesize=ccd_size,
            target_device_idx=target_device_idx
        )
        self.pupdata.set_slopes_from_intensity()

        # Extract valid indices for both pupils using pupdata's xp object
        # (Needed before calling super().__init__ which calls nslopes)
        all_idx = self.pupdata.pupil_idx(0).astype(self.pupdata.xp.int64)
        self.pup_idx = all_idx[all_idx >= 0]

        super().__init__(sn=sn, interleave=interleave,
                         target_device_idx=target_device_idx, precision=precision)

        # Modify Inputs: CWFS requires two images (Intra/Extra focal)
        if 'in_pixels' in self.inputs:
            del self.inputs['in_pixels']

        self.inputs['in_pixels1'] = InputValue(type=Pixels)
        self.inputs['in_pixels2'] = InputValue(type=Pixels)

        # Outputs to track the used pupils
        self.outputs['out_pupdata'] = self.pupdata

        # Setup slopes display mapping (using the first pupdata as geometric reference)
        self.slopes.single_mask = self.pupdata.single_mask()
        self.slopes.display_map = self.pupdata.display_map

        self.flat_p1 = None
        self.flat_p2 = None

    @classmethod
    def input_names(cls):
        return {'in_pixels1': InputDesc(Pixels, 'Intra-focal pixel image from detector'),
                'in_pixels2': InputDesc(Pixels, 'Extra-focal pixel image from detector')}

    @classmethod
    def output_names(cls):
        result = super().output_names()
        result.update({
            'out_pupdata': OutputDesc(PupData, 'Pupil data with subaperture geometry')
        })
        return result

    def nsubaps(self):
        # Every pixel is treated as a subaperture
        return len(self.pup_idx)

    def nslopes(self):
        # 1 signal (curvature) per valid pixel
        return len(self.pup_idx)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        # Flatten incoming pixel arrays
        self.flat_p1 = self.local_inputs['in_pixels1'].pixels.flatten()
        self.flat_p2 = self.local_inputs['in_pixels2'].pixels.flatten()

    def trigger_code(self):
        # Extract valid pixels according to PupData
        i1 = self.flat_p1[self.pup_idx].astype(self.xp.float32)
        i2 = self.flat_p2[self.pup_idx].astype(self.xp.float32)

        # Compute Curvature Signal: S = (I1 - I2) / (I1 + I2)
        sum_i = i1 + i2

        # Avoid division by zero
        sum_i = self.xp.where(sum_i < 1e-9, 1.0, sum_i)

        signal = (i1 - i2) / sum_i

        # Store results
        self.slopes.slopes[:] = signal

        # Update telemetry
        self.flux_per_subaperture_vector.value[:] = sum_i
        total_int = self.xp.sum(sum_i)
        self.total_counts.value[0] = total_int
        self.subap_counts.value[0] = total_int / self.nsubaps()

    def post_trigger(self):
        super().post_trigger()
        self.outputs['out_pupdata'].generation_time = self.current_time
