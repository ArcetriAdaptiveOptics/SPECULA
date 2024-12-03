
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes


class Slopec(BaseProcessingObj):
    def __init__(self, sn: Slopes=None, cm=None, 
                 use_sn=False, accumulate=False, weight_from_accumulated=False, 
                 do_rec=False, 
                 intmat=None, 
                 recmat=None, accumulation_dt=0, 
                 accumulated_pixels=(0,0),
                 target_device_idx=None, 
                 precision=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.slopes = Slopes(2)  # TODO resized in derived class
        self.sn = sn
        self.cm = cm
        self.flux_per_subaperture_vector = BaseValue()
        self.max_flux_per_subaperture_vector = BaseValue()
        self.use_sn = use_sn
        self.accumulate = accumulate
        self.weight_from_accumulated = weight_from_accumulated
        self.do_rec = do_rec
        self.intmat = intmat
        self.recmat = recmat
        self._accumulation_dt = accumulation_dt
        self.accumulated_pixels = self.xp.array(accumulated_pixels, dtype=self.dtype)
        self.accumulated_slopes = Slopes(2)   # TODO resized in derived class.

        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.inputs['in_pixels_list'] = InputValue(type=Pixels)
        self.outputs['out_slopes'] = self.slopes


    @property
    def sn_tag(self):
        return self._sn_tag

    @sn_tag.setter
    def sn_tag(self, value):
        self.load_sn(value)

    @property
    def accumulation_dt(self):
        return self._accumulation_dt

    @accumulation_dt.setter
    def accumulation_dt(self, value):
        self._accumulation_dt = self.seconds_to_t(value)

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
        # super().post_trigger()
        if self.do_rec:
            m = self.xp.dot(self.slopes.slopes, self.recmat.recmat)
            self.slopes.slopes = m

    def run_check(self, time_step, errmsg=''):
        self.prepare_trigger(0)
        if self.use_sn and not self.sn:
            errmsg += 'Slopes null are not valid'
        if self.weight_from_accumulated and self.accumulate:
            errmsg += 'weightFromAccumulated and accumulate must not be set together'
        if errmsg != '':
            print(errmsg)
        pixels_ok = self.local_inputs['in_pixels'] or self.local_inputs['in_pixels_list']
        return not (self.weight_from_accumulated and self.accumulate) and pixels_ok and self.slopes and ((not self.use_sn) or (self.use_sn and self.sn))

