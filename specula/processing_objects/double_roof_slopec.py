from specula.processing_objects.pyr_slopec import PyrSlopec
from specula.base_processing_obj import InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.pupdata import PupData
from specula.data_objects.slopes import Slopes


class DoubleRoofSlopec(PyrSlopec):
    '''
    Double roof slope computer processing object.
    A DoubleRoofSlopec is a standard pyramid slope computer,
    customized for the double-roof case.
    '''
    def __init__(self,
                 pupdata: PupData,
                 sn: Slopes=None,
                 shlike: bool=False,
                 norm_factor: float=None,   # TODO =1.0,
                 thr_value: float=0,
                 slopes_from_intensity: bool=False,
                 target_device_idx: int=None,
                 precision: int=None,
                **kwargs): # is this needed??

        super().__init__(pupdata=pupdata,
                        sn=sn,
                        shlike=shlike,
                        norm_factor=norm_factor,
                        thr_value=thr_value,
                        slopes_from_intensity=slopes_from_intensity,
                        target_device_idx=target_device_idx,
                        precision=precision,
                        **kwargs)

    @classmethod
    def input_names(cls):
        return {'in_pixels': InputDesc(Pixels, 'Input pixel data from the double-roof wavefront sensor detector')}

    @classmethod
    def output_names(cls):
        return {'out_slopes': OutputDesc(Slopes, 'Computed wavefront slopes'),
                'out_flux_per_subaperture': OutputDesc(BaseValue, 'Flux per subaperture'),
                'out_total_counts': OutputDesc(BaseValue, 'Total photon counts'),
                'out_subap_counts': OutputDesc(BaseValue, 'Counts per subaperture'),
                'out_pupdata': OutputDesc(PupData, 'Pupil data with subaperture geometry')}

    def _compute_pyr_slopes(self, A, B, C, D, factor):

        # DOUBLE ROOF SLOPE CALCULATION:
        # When axis origin is in bottom-left corner:
        # - A is bottom-right
        # - B is bottom-left
        # - C is top-left
        # - D is top-right
        # So, A - B and D - C gives the same sign
        sx = (A - B) * factor  # roof2 horizontal separation
        sy = (D - C) * factor  # roof1 vertical separation after shift
        return sx, sy
