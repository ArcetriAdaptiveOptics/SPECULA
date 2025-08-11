import numpy as np
from specula import show_in_profiler

from specula.base_processing_obj import BaseProcessingObj
from specula.base_data_obj import BaseDataObj
from specula.base_value import BaseValue
from specula.data_objects.layer import Layer
from specula.connections import InputValue
from specula import cpuArray, ASEC2RAD, RAD2ASEC
from specula.data_objects.simul_params import SimulParams

from seeing.sympyHelpers import *
from seeing.formulary import *
from seeing.integrator import *


from symao.turbolence import createTurbolenceFormulary, ft_phase_screen0, ft_ft2

turbolenceFormulas = createTurbolenceFormulary()

def seeing_to_r0(seeing, wvl=500.e-9):
    return 0.9759*wvl/(seeing* ASEC2RAD)

def cn2_to_r0(cn2, wvl=500.e-9):
    r0=(0.423*(2*np.pi/wvl)**2*cn2)**(-3./5.)
    return r0

def r0_to_seeing(r0, wvl=500.e-9):
    return (0.9759*wvl/r0)*RAD2ASEC

def cn2_to_seeing(cn2, wvl=500.e-9):
    r0 = cn2_to_r0(cn2,wvl)
    seeing = r0_to_seeing(r0,wvl)
    return seeing


class InfinitePhaseScreen(BaseDataObj):

    def __init__(self, mx_size, pixel_scale, r0, L0, l0, random_seed=None, stencil_size_factor=1, xp=np, target_device_idx=0, precision=0):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.random_data_col = None
        self.random_data_row = None
        self.requested_mx_size = int(mx_size)
        self.mx_size = 2 ** (int( np.ceil(np.log2(mx_size)))) + 1
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.l0 = l0
        self.xp = xp
        self.stencil_size_factor = stencil_size_factor
        self.stencil_size = stencil_size_factor * self.mx_size
        if random_seed is None:
            raise ValueError("random_seed must be provided")
        else:
            self.random_seed = int(random_seed)
        self.rng = self.xp.random.default_rng(self.random_seed)

        #self.set_stencil_coords_basic()
        self.set_stencil_coords()
        self.setup()

    def phase_covariance(self, r, r0, L0):
        r = cpuArray(r)
        r0 = float(r0)
        L0 = float(L0)
        # Get rid of any zeros
        r += 1e-40
        exprCf = turbolenceFormulas['phaseVarianceVonKarman0'].rhs
        (_, cov) = evaluateFormula( exprCf, {'r_0': r0, 'L_0': L0}, ['r'] , [r], cpulib)

#        A = (L0 / r0) ** (5. / 3)
#        B1 = (2 ** (-5. / 6)) * gamma(11. / 6) / (self.xp.pi ** (8. / 3))
#        B2 = ((24. / 5) * gamma(6. / 5)) ** (5. / 6)
#        C = (((2 * self.xp.pi * r) / L0) ** (5. / 6)) * kv(5. / 6, (2 * self.xp.pi * r) / L0)
#        cov = A * B1 * B2 * C / 2

        cov = self.xp.asarray(cov)

        return cov

    def set_stencil_coords_basic(self):
        self.stencil = self.xp.zeros((self.stencil_size, self.stencil_size))
        self.stencil[:2,:] = 1
        self.stencil_coords = self.to_xp(self.xp.where(self.stencil==1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale
        self.n_stencils = self.stencil_coords.shape[0]

    def set_stencil_coords(self):
        self.stencil = np.zeros((self.stencil_size, self.stencil_size))
        self.stencilF = np.zeros((self.stencil_size, self.stencil_size))
        max_n = int( np.floor(np.log2(self.stencil_size)))
        # the head of stencil (basiaccaly all of it for us)
        for n in range(0, max_n + 1):
            col = int((2 ** (n - 1)) + 1)
            n_points = (2 ** (max_n - n)) + 1
            coords = np.round(np.linspace(0, self.stencil_size - 1, n_points)).astype('int32')
            self.stencil[col - 1][coords] = 1
            self.stencilF[self.stencil_size - col][coords] = 1
        # the tail of stencil
        for n in range(1, self.stencil_size_factor + 1):
            col = n * self.mx_size - 1
            self.stencil[col, self.stencil_size // 2] = 1
            self.stencilF[self.stencil_size-col-1, self.stencil_size // 2] = 1
        self.stencil = self.xp.asarray(self.stencil)
        self.stencilF = self.xp.asarray(self.stencilF)
        self.stencil_coords = []
        self.stencil_coords.append(self.to_xp(self.xp.where(self.stencil == 1)).T)
        self.stencil_coords.append(self.to_xp(self.xp.where(self.stencilF == 1)).T)
        self.stencil_positions = []
        self.stencil_positions.append(self.stencil_coords[0] * self.pixel_scale)
        self.stencil_positions.append(self.stencil_coords[1] * self.pixel_scale)        
        self.n_stencils = self.stencil_coords[0].shape[0]

    def AB_from_positions(self, positions):
        seperations = self.xp.zeros((len(positions), len(positions)))
        px, py = positions[:,0], positions[:,1]
        delta_x_gridA, delta_x_gridB = self.xp.meshgrid(px, px)
        delta_y_gridA, delta_y_gridB = self.xp.meshgrid(py, py)
        delta_x_grid = delta_x_gridA - delta_x_gridB
        delta_y_grid = delta_y_gridA - delta_y_gridB
        seperations = self.xp.sqrt(delta_x_grid ** 2 + delta_y_grid ** 2)
        self.cov_mat = self.phase_covariance(seperations, self.r0, self.L0)
        self.cov_mat_zz = self.cov_mat[:self.n_stencils, :self.n_stencils]
        self.cov_mat_xx = self.cov_mat[self.n_stencils:, self.n_stencils:]
        self.cov_mat_zx = self.cov_mat[:self.n_stencils, self.n_stencils:]
        self.cov_mat_xz = self.cov_mat[self.n_stencils:, :self.n_stencils]
        # Cholesky solve can fail - so do brute force inversion
        cf = self._lu_factor(self.cov_mat_zz)
        inv_cov_zz = self._lu_solve(cf, self.xp.identity(self.cov_mat_zz.shape[0]))
        A_mat = self.cov_mat_xz.dot(inv_cov_zz)
        # Can make initial BBt matrix first
        BBt = self.cov_mat_xx - A_mat.dot(self.cov_mat_zx)
        # Then do SVD to get B matrix
        u, W, ut = self.xp.linalg.svd(BBt)
        L_mat = self.xp.zeros((self.stencil_size, self.stencil_size))
        self.xp.fill_diagonal(L_mat, self.xp.sqrt(W))
        # Now use sqrt(eigenvalues) to get B matrix
        B_mat = u.dot(L_mat)
        return A_mat, B_mat
    
    def setup(self):
        # set X coords
        self.new_col_coords1 = self.xp.zeros((self.stencil_size, 2))
        self.new_col_coords1[:, 0] = -1
        self.new_col_coords1[:, 1] = self.xp.arange(self.stencil_size)
        self.new_col_positions1 = self.new_col_coords1 * self.pixel_scale
        # calc separations
        positions1 = self.xp.concatenate((self.stencil_positions[0], self.new_col_positions1), axis=0)
        self.A_mat, self.B_mat = [], []
        A_mat, B_mat = self.AB_from_positions(positions1)
        self.A_mat.append(A_mat)
        self.B_mat.append(B_mat)
        self.A_mat.append(self.xp.fliplr(self.xp.flipud(A_mat)))
        self.B_mat.append(B_mat)
        # make initial screen
        tmp, _, _ = ft_phase_screen0( turbolenceFormulas, self.r0, self.stencil_size, self.pixel_scale, self.L0, seed=self.random_seed)
        self.full_scrn = self.xp.asarray(tmp) / 2
        self.full_scrn -= self.xp.mean(self.full_scrn)
        # print(self.full_scrn.shape)

    def prepare_random_data_col(self):
        if self.random_data_col is None:
#            print('generating new random data col')
            self.random_data_col = self.rng.standard_normal(size=self.stencil_size)
        else:
            pass
#            print('using old random data col')

    def prepare_random_data_row(self):
        if self.random_data_row is None:
#            print('generating new random data row')
            self.random_data_row = self.rng.standard_normal(size=self.stencil_size)
        else:
            pass
#            print('using old random data row')

    def get_new_line(self, row, after):
        if row:
            self.prepare_random_data_row()
            stencil_data = self.xp.asarray(self.full_scrn[self.stencil_coords[after][:, 1], self.stencil_coords[after][:, 0]])
            new_line = self.A_mat[after].dot(stencil_data) + self.B_mat[after].dot(self.random_data_row)  
        else:
            self.prepare_random_data_col()
            stencil_data = self.xp.asarray(self.full_scrn[self.stencil_coords[after][:, 0], self.stencil_coords[after][:, 1]])            
            new_line = self.A_mat[after].dot(stencil_data) + self.B_mat[after].dot(self.random_data_col)
        return new_line

    def add_line(self, row, after, flush=True):
        new_line = self.get_new_line(row, after)
        if row:
            new_line = new_line[:,self.xp.newaxis]
            if after:
                self.full_scrn = self.xp.concatenate((self.full_scrn, new_line), axis=row)[:self.stencil_size, 1:]
            #    self.shift(self.full_scrn, [-1, 0], self.full_scrn, order=0, mode='constant', cval=0.0, prefilter=False)
            #    self.full_scrn[-1, :] = new_line
            else:
                self.full_scrn = self.xp.concatenate((new_line, self.full_scrn), axis=row)[:self.stencil_size, :self.stencil_size]
            #    self.shift(self.full_scrn, [1, 0], self.full_scrn, order=0, mode='constant', cval=0.0, prefilter=False)
            #    self.full_scrn[0, :] = new_line
        else:
            new_line = new_line[self.xp.newaxis, :]
            if after:
                self.full_scrn = self.xp.concatenate((self.full_scrn, new_line), axis=row)[1:, :self.stencil_size]
            #    self.shift(self.full_scrn, [0, -1], self.full_scrn, order=0, mode='constant', cval=0.0, prefilter=False)
            #    self.full_scrn[:, -1] = new_line
            else:
                self.full_scrn = self.xp.concatenate((new_line, self.full_scrn), axis=row)[:self.stencil_size, :self.stencil_size]
            #    self.shift(self.full_scrn, [0, 1], self.full_scrn, order=0, mode='constant', cval=0.0, prefilter=False)
            #    self.full_scrn[:, 0] = new_line
        if flush:
            self.random_data_col = None
            self.random_data_row = None

    @property
    def scrn(self):
        return cpuArray(self.full_scrn[:self.requested_mx_size, :self.requested_mx_size])

    @property
    def scrnRaw(self):
        return self.full_scrn[:self.requested_mx_size, :self.requested_mx_size]

    @property
    def scrnRawAll(self):
        return self.full_scrn


class AtmoInfiniteEvolution(BaseProcessingObj):
    def __init__(self,
                 simul_params: SimulParams,
                 L0: list=[1.0],
                 heights: list=[0.0],
                 Cn2: list=[1.0],
                 zenithAngleInDeg: float=0.0,
                 fov: float=0.0,
                 seed: int=1,
                 extra_delta_time: float=0,
                 verbose: bool=False,
                 fov_in_m: float=None,
                 pupil_position:list =[0,0],
                 target_device_idx: int=None,
                 precision: int=None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params

        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch

        self.n_infinite_phasescreens = len(heights)
        self.last_position = np.zeros(self.n_infinite_phasescreens)
        self.last_t = 0
        self.delta_time = None
        # fixed at generation time, then is a input -> rescales the screen?
        self.seeing = 1.0
        self.l0 = 0.005
        self.airmass = 1
        self.ref_wavelengthInNm = 500
        self.extra_delta_time = extra_delta_time

        self.inputs['seeing'] = InputValue(type=BaseValue)
        self.inputs['wind_speed'] = InputValue(type=BaseValue)
        self.inputs['wind_direction'] = InputValue(type=BaseValue)

        if pupil_position is None:
            pupil_position = [0, 0]

        if zenithAngleInDeg is not None:
            self.airmass = 1.0 / np.cos(np.radians(zenithAngleInDeg), dtype=self.dtype)
            print(f'Atmo_Evolution: zenith angle is defined as: {zenithAngleInDeg} deg')
            print(f'Atmo_Evolution: airmass is: {self.airmass}')
        else:
            self.airmass = np.array(1.0, dtype=self.dtype)
        self.heights = np.array(heights, dtype=self.dtype) * self.airmass

        alpha_fov = fov / 2.0

        # Max star angle from arcseconds to radians
        rad_alpha_fov = alpha_fov * ASEC2RAD

        # Compute layers dimension in pixels
        self.pixel_layer_size = np.ceil((self.pixel_pupil + 2 * np.sqrt(np.sum(np.array(pupil_position, dtype=self.dtype) * 2)) / self.pixel_pitch + 
                               2.0 * abs(self.heights) / self.pixel_pitch * rad_alpha_fov) / 2.0) * 2.0
        if fov_in_m is not None:
            self.pixel_layer_size = np.full_like(self.heights, int(fov_in_m / self.pixel_pitch / 2.0) * 2)

        self.L0 = L0

        if np.isscalar(self.L0):
            self.L0 = [self.L0] * len(self.heights)
        elif len(self.L0) != len(self.heights):
            raise ValueError(f"L0 must have the same length as heights ({len(self.heights)}), got {len(self.L0)}")

        self.Cn2 = np.array(Cn2, dtype=self.dtype)
        self.verbose = verbose if verbose is not None else False

        # Initialize layer list with correct heights
        self.layer_list = []
        for i in range(self.n_infinite_phasescreens):
            layer = Layer(self.pixel_layer_size[i], self.pixel_layer_size[i], self.pixel_pitch, self.heights[i], precision=self.precision, target_device_idx=self.target_device_idx)
            self.layer_list.append(layer)
        self.outputs['layer_list'] = self.layer_list

        self.initScreens(seed)

        self.last_position = np.zeros(self.n_infinite_phasescreens, dtype=self.dtype)
        self.scale_coeff = 1.0

        if not np.isclose(np.sum(self.Cn2), 1.0, atol=1e-6):
            raise ValueError(f' Cn2 total must be 1. Instead is: {np.sum(self.Cn2)}.')

    def initScreens(self, seed):
        self.seed = seed
        if self.seed <= 0:
            raise ValueError('seed must be >0')
        # Phase screens list
        self.infinite_phasescreens = []
        seed = self.seed + self.xp.arange(self.n_infinite_phasescreens)
        if len(seed) != len(self.L0):
            raise ValueError('Number of elements in seed and L0 must be the same!')

        self.acc_rows = np.zeros((self.n_infinite_phasescreens))
        self.acc_cols = np.zeros((self.n_infinite_phasescreens))

        # Square infinite_phasescreens
        for i in range(self.n_infinite_phasescreens):
            print('Creating phase screen..')
            self.ref_r0 = 0.9759 * 0.5 / (self.seeing * 4.848) * self.airmass**(-3./5.) # if seeing > 0 else 0.0
#            self.ref_r0 *= (self.ref_wavelengthInNm / 500.0 / ((2*np.pi)))**(6./5.)
            self.ref_r0 *= (self.ref_wavelengthInNm / 500.0 )**(6./5.)
            print('self.ref_r0:', self.ref_r0)
            temp_infinite_screen = InfinitePhaseScreen(self.pixel_layer_size[i],
                                                       self.pixel_pitch,
                                                       self.ref_r0,
                                                       self.L0[i],
                                                       self.l0,
                                                       random_seed=int(seed[i]),
                                                       xp=self.xp,
                                                       target_device_idx=self.target_device_idx,
                                                       precision=self.precision )
            self.infinite_phasescreens.append(temp_infinite_screen)

    def setup(self):
        super().setup()
        # check that seeing is a 1-element array
        if len(self.local_inputs['seeing'].value) != 1:
            raise ValueError('Seeing input must be a 1-element array')

        # Check that wind speed and direction have the correct length
        if len(self.local_inputs['wind_speed'].value) != self.n_infinite_phasescreens:
            raise ValueError('Wind speed input must be a {self.n_infinite_phasescreens}-elements array')
        if len(self.local_inputs['wind_direction'].value) != self.n_infinite_phasescreens:
            raise ValueError('Wind direction input must be a {self.n_infinite_phasescreens}-elements array')

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.delta_time = self.t_to_seconds(self.current_time - self.last_t) + self.extra_delta_time
        seeing = float(cpuArray(self.local_inputs['seeing'].value[0]))

        if seeing > 0:
            r0 = 0.9759 * 0.5 / (seeing * 4.848) * self.airmass**(-3./5.)
            r0 *= (self.ref_wavelengthInNm / 500)**(6./5.)
            scale_r0 = (self.ref_r0 / r0)**(5./6.)
        else:
            scale_r0 = 0.0

        scale_wvl = ( self.ref_wavelengthInNm / (2 * np.pi) )
        self.scale_coeff = scale_r0 * scale_wvl

    @show_in_profiler('atmo_evolution.trigger_code')
    def trigger_code(self):
        wind_speed = cpuArray(self.local_inputs['wind_speed'].value)
        wind_direction = cpuArray(self.local_inputs['wind_direction'].value)

        # Compute the delta position in pixels
        delta_position =  wind_speed * self.delta_time / self.pixel_pitch  # [pixel]
        new_position = self.last_position + delta_position
        eps = 1e-4

        for ii, phaseScreen in enumerate(self.infinite_phasescreens):
            w_y_comp = np.cos(2*np.pi*(wind_direction[ii])/360.0)
            w_x_comp = np.sin(2*np.pi*(wind_direction[ii])/360.0)
            frac_rows, rows_to_add = np.modf( delta_position[ii] * w_y_comp + self.acc_rows[ii])
            #sr = int( (np.sign(rows_to_add) + 1) / 2 )
            sr = int(np.sign(rows_to_add) )
            frac_cols, cols_to_add = np.modf( delta_position[ii] * w_x_comp + self.acc_cols[ii] )
            #sc = int( (-np.sign(cols_to_add) + 1) / 2 )
            sc = int(np.sign(cols_to_add) )
            # print('rows_to_add, cols_to_add', rows_to_add, cols_to_add)
            if np.abs(w_y_comp)>eps:
                for r in range(int(np.abs(rows_to_add))):
                    phaseScreen.add_line(1, sr)
            if np.abs(w_x_comp)>eps:
                for r in range(int(np.abs(cols_to_add))):
                    phaseScreen.add_line(0, sc)
            phaseScreen0All = phaseScreen.scrnRawAll.copy()
            phaseScreen0 = phaseScreen.scrnRaw.copy()
            # print('w_y_comp, w_x_comp', w_y_comp, w_x_comp)
            # print('frac_rows, frac_cols', frac_rows, frac_cols)
            srf = int(np.sign(frac_rows) )
            scf = int(np.sign(frac_cols) )

            if np.abs(frac_rows)>eps:
                phaseScreen.add_line(1, srf, False)
            if np.abs(frac_cols)>eps:
                phaseScreen.add_line(0, scf, False)
            phaseScreen1 = phaseScreen.scrnRaw
            interpfactor = np.sqrt(frac_rows**2 + frac_cols**2 )
            layer_phase = interpfactor * phaseScreen1 + (1.0-interpfactor) * phaseScreen0
            phaseScreen.full_scrn = phaseScreen0All
            self.acc_rows[ii] = frac_rows
            self.acc_cols[ii] = frac_cols
            # print('acc_rows', self.acc_rows)
            # print('acc_cols', self.acc_cols)
            self.layer_list[ii].field[:] = self.xp.stack((layer_phase, layer_phase))
            self.layer_list[ii].phaseInNm *= self.scale_coeff
            self.layer_list[ii].A = 1
            self.layer_list[ii].generation_time = self.current_time
        self.last_position = new_position
        self.last_t = self.current_time
