
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat


class Slopec(BaseProcessingObj):
    def __init__(self,
                 sn: Slopes=None, 
                 use_sn: bool=False,
                 accumulate: bool=False,
                 weight_from_accumulated: bool=False,
                 recmat: Recmat=None,
                 filt_intmat: Intmat=None, 
                 filt_recmat: Recmat=None,
                 filtmat=None,
                 accumulation_dt: float=0, 
                 accumulated_pixels: tuple=(0,0),
                 target_device_idx: int=None, 
                 precision: int=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # TODO this can become a single parameter (no need for separate flag)
        if use_sn and not sn:
            raise ValueError('Slopes null are not valid')
        
        # TODO this can become a single parameter (no need for separate flag)
        if weight_from_accumulated and accumulate:
            raise ValueError('weightFromAccumulated and accumulate must not be set together')

        self.slopes = Slopes(2)  # TODO resized in derived class
        self.sn = sn
        self.flux_per_subaperture_vector = BaseValue()
        self.max_flux_per_subaperture_vector = BaseValue()
        self.use_sn = use_sn
        self.accumulate = accumulate
        self.weight_from_accumulated = weight_from_accumulated
        self.recmat = recmat
        if filtmat is not None:
            if filt_intmat:
                raise ValueError('filt_intmat must not be set if "filtmat" is set')
            if filt_recmat:
                raise ValueError('filt_recmat must not be set if "filtmat" is set')
            self.filt_intmat = Intmat(filtmat[0], target_device_idx=target_device_idx)
            self.filt_recmat = Recmat(filtmat[1], target_device_idx=target_device_idx)
        else:
            if bool(filt_intmat) != bool(filt_recmat):
                raise ValueError('Both filt_intmat and filt_recmat must be set for slopes filtering')
            self.filt_intmat = filt_intmat
            self.filt_recmat = filt_recmat

        self.accumulation_dt = self.seconds_to_t(accumulation_dt)
        self.accumulated_pixels = self.xp.array(accumulated_pixels, dtype=self.dtype)
        self.accumulated_slopes = Slopes(2)   # TODO resized in derived class.

        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.outputs['out_slopes'] = self.slopes

    @property
    def sn_tag(self):
        return self._sn_tag

    @sn_tag.setter
    def sn_tag(self, value):
        self.load_sn(value)

    def build_and_save_filtmat(self, intmat, recmat, nmodes, filename):
        im = intmat[:nmodes, :]
        rm = recmat[:, :nmodes]

        output = self.xp.stack((im, self.xp.transpose(rm)), axis=-1)
        self.writefits(filename, output)
        print(f'saved {filename}')

    def _compute_flux_per_subaperture(self):
        raise NotImplementedError('abstract method must be implemented')

    def _compute_max_flux_per_subaperture(self):
        raise NotImplementedError('abstract method must be implemented')

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

    def trigger_code(self):
        raise NotImplementedError(f'{self.__class__.__name__}: please implement trigger_code() in your derived class!')
        
    def post_trigger(self):
        if self.recmat:
            m = self.xp.dot(self.slopes.slopes, self.recmat.recmat)
            self.slopes.slopes = m
        
        if self.filt_intmat and self.filt_recmat:
            m = self.slopes.slopes @ self.filt_recmat.recmat
            sl0 = m @ self.filt_intmat.intmat.T
            self.slopes.slopes -= sl0
            rms = self.xp.sqrt(self.xp.mean(self.slopes.slopes**2))
            #print('Slopes have been filtered. '
            #      'New slopes min, max and rms: '
            #      f'{self.slopes.slopes.min()}, {self.slopes.slopes.max()}, {rms}')

