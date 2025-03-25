from specula.base_processing_obj import BaseProcessingObj

class ShShift(BaseProcessingObj):
    '''
    TODO not yet working
    '''
    def __init__(self, params_sh, params_main, shift_wavelength_in_nm, xy_shift, qe_factor, resize_fact, 
                 target_device_idx=None, 
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        raise NotImplementedError

        self._params_sh = params_sh
        self._shift_wavelength_in_nm = self.xp.array(shift_wavelength_in_nm)
        self._xy_shift = self.xp.array(xy_shift)
        self._qe_factor = self.xp.array(qe_factor)
        self._resize_fact = resize_fact

        self._n_shift = self._xy_shift.shape[1]
        self._ccd_side = self._params_sh.subap_on_diameter * self._params_sh.sensor_npx
        self._out_i = self.xp.zeros((self._ccd_side, self._ccd_side))

        if self._shift_wavelength_in_nm.size != self._n_shift:
            raise ValueError("SH Shift: xyShift and wavelength vector must have the same size")
        if self._qe_factor.size != self._n_shift:
            raise ValueError("SH Shift: wavelength and qe factor array must have the same size")

        self.do_factory(params_main)

    def do_factory(self, params_main):
        # Build the objects in function of the shift
        factory = Factory(params_main, GPU=self._GPU)

        self._ef_shift = self.xp.empty(self._n_shift, dtype=object)
        self._ef_resize = self.xp.empty(self._n_shift, dtype=object)
        self._sh = self.xp.empty(self._n_shift, dtype=object)

        params_ef_shift = {}
        params_ef_resize = {'resize_fact': self._resize_fact}

        for i_shift in range(self._n_shift):
            params_ef_shift['xyShift'] = self._xy_shift[:, i_shift]
            params_sh_tmp = self._params_sh.copy()
            params_sh_tmp.wavelength_in_nm = self._shift_wavelength_in_nm[i_shift]

            self._sh[i_shift] = factory.get_sh(params_sh_tmp, GPU=self._GPU)
            self._ef_shift[i_shift] = factory.get_ef_shift(params_ef_shift)
            self._ef_resize[i_shift] = factory.get_ef_resize(params_ef_resize)

    @property
    def shift_wavelength_in_nm(self):
        return self._shift_wavelength_in_nm

    @property
    def xy_shift(self):
        return self._xy_shift

    @property
    def qe_factor(self):
        return self._qe_factor

    @property
    def resize_fact(self):
        return self._resize_fact

    @property
    def in_ef(self):
        return self._in_ef

    @in_ef.setter
    def in_ef(self, in_ef):
        self._in_ef = in_ef
        for i_shift in range(self._n_shift):
            self._ef_shift[i_shift].in_ef = self._in_ef
            self._ef_resize[i_shift].in_ef = self._ef_shift[i_shift].out_ef
            self._sh[i_shift].in_ef = self._ef_resize[i_shift].out_ef

    def trigger(self, t):
        if self._in_ef.generation_time != t:
            return
        
        self._out_i.fill(0.0)  # Reset the output image
        
        for i_shift in range(self._n_shift):
            self._ef_shift[i_shift].trigger(t)
            self._ef_resize[i_shift].trigger(t)
            self._sh[i_shift].trigger(t)
            self._out_i += self._sh[i_shift].out_i * self._qe_factor[i_shift]

        self._out_i_generation_time = t

    def revision_track(self):
        return '$Rev$'

    def cleanup(self):
        self._in_ef = None
        self._out_i = None
        self._params_sh = None
        self._shift_wavelength_in_nm = None
        self._xy_shift = None
        self._qe_factor = None
        self._ef_shift = None
        self._ef_resize = None
        self._sh = None

# Helper classes for the functionality (e.g., `Factory`, `BaseProcessingObj`)
class Factory:
    def __init__(self, params_main, GPU):
        # Assume this initializes a factory for SH, EF objects
        pass

    def get_sh(self, params_sh, GPU):
        # Return an SH object based on params_sh and GPU
        pass

    def get_ef_shift(self, params_ef_shift):
        # Return an EF shift object
        pass

    def get_ef_resize(self, params_ef_resize):
        # Return an EF resize object
        pass

class BaseProcessingObj:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def cleanup(self):
        # Basic cleanup functionality
        pass
