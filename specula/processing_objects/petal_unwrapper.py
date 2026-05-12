from specula import np, cpuArray
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.ifunc import IFunc
from specula.data_objects.pupilstop import Pupilstop

class PetalUnwrapper(BaseProcessingObj):
    """
    Topological Unwrapper for Segmented Mirrors.
    Agnostic version: Always computes 3*N_petals DOFs internally to absorb atmospheric tilt 
    robustly, but can filter the output to actuate Pistons only.
    """
    def __init__(self,
                 ifunc: IFunc,
                 pupilstop: Pupilstop,
                 n_petals: int = 6,
                 angle_offset_deg: float = 30.0,
                 spider_widths: list = None,
                 thresh_in_nm: float = 350.0,
                 nmodes: int = None,
                 start_time: float = 0.0,
                 interval_time: float = 0.0,
                 rcond: float = 1e-2,
                 piston_only: bool = False,
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.n_petals = n_petals
        self.angle_offset_deg = angle_offset_deg
        self.thresh_in_nm = thresh_in_nm
        self.nmodes = nmodes
        self.rcond = rcond
        self.piston_only = piston_only

        # SPECULA discrete time management
        self.start_time_t = self.seconds_to_t(start_time)
        self.interval_time_t = self.seconds_to_t(interval_time)
        self.last_correction_time = -1

        if spider_widths is None:
            self.spider_widths = [0.5] * self.n_petals
        else:
            self.spider_widths = spider_widths

        self.inputs['in_comm'] = InputValue(type=BaseValue)

        self.outputs['out_comm'] = BaseValue(target_device_idx=self.target_device_idx,
                                             precision=self.precision)
        self.outputs['out_ost'] = BaseValue(target_device_idx=self.target_device_idx,
                                            precision=self.precision)

        self._initialize_geometry(ifunc, pupilstop)

    @classmethod
    def input_names(cls):
        return {'in_comm': InputDesc(BaseValue, 'Input command vector from integrators')}

    @classmethod
    def output_names(cls):
        return {
            'out_comm': OutputDesc(BaseValue, 'Cleaned command vector for the DM'),
            'out_ost': OutputDesc(BaseValue, 'State correction vector for the IirFilter')
        }

    def _initialize_geometry(self, ifunc, pupilstop):
        self.logger.info(f"Generating internal Petal modes ({self.n_petals * 3} DOFs) and mapping to basis...")

        mask_amp = cpuArray(pupilstop.A) > 0
        dim = mask_amp.shape[0]
        pitch = pupilstop.pixel_pitch

        x_1d = (np.arange(dim) - dim / 2.0 + 0.5) * pitch
        X, Y = np.meshgrid(x_1d, x_1d)
        R = np.sqrt(X**2 + Y**2)
        Theta = np.degrees(np.arctan2(Y, X)) % 360.0

        # 1. ALWAYS GENERATE 3*N_petals IDEAL PETALS (Piston, Tip, Tilt)
        # This is strictly required to let the math absorb continuous atmospheric tilts
        n_valid = np.sum(mask_amp)
        self.n_ideal_modes = self.n_petals * 3
        P_ideal = np.zeros((self.n_ideal_modes, n_valid), dtype=np.float32)

        pupil_radius = (dim / 2.0) * pitch # For normalization

        for i in range(self.n_petals):
            ang_start = (self.angle_offset_deg + i * (360.0 / self.n_petals)) % 360.0
            ang_end = (self.angle_offset_deg + (i + 1) * (360.0 / self.n_petals)) % 360.0

            if ang_start < ang_end:
                sector_mask = (Theta >= ang_start) & (Theta < ang_end)
            else: # Wraparound across 0 degrees
                sector_mask = (Theta >= ang_start) | (Theta < ang_end)

            sector_1d = sector_mask[mask_amp].astype(np.float32)

            # Mode 0: Piston
            P_ideal[3*i + 0, :] = sector_1d
            # Mode 1: Tip (X-Tilt)
            P_ideal[3*i + 1, :] = sector_1d * (X[mask_amp] / pupil_radius)
            # Mode 2: Tilt (Y-Tilt)
            P_ideal[3*i + 2, :] = sector_1d * (Y[mask_amp] / pupil_radius)

        # 1b. Prepare the surgical mask for "piston_only" mode
        piston_mask_cpu = np.zeros(self.n_ideal_modes, dtype=np.float32)
        piston_mask_cpu[0::3] = 1.0  # Keep indices 0, 3, 6... (Pistons), kill Tips and Tilts
        self.piston_mask = self.to_xp(piston_mask_cpu, dtype=self.dtype)

        # 2. PROJECT IDEAL PETALS ONTO THE PROVIDED BASIS
        B_full = cpuArray(ifunc.influence_function) # Shape: (n_modes, n_valid)

        if self.nmodes is not None:
            B = B_full[:self.nmodes, :]
        else:
            B = B_full

        n_modes = B.shape[0]

        BBT = B @ B.T
        BBT_pinv = np.linalg.pinv(BBT, rcond=1e-4)
        C_petals_cpu = BBT_pinv @ B @ P_ideal.T # Shape: (n_modes, 18)
        self.C_petals = self.to_xp(C_petals_cpu, dtype=self.dtype)

        # 3. BUILD FULL OBSERVATION MATRIX (H_full)
        # Using the robust 1D radial regression.
        H_full_cpu = np.zeros((self.n_petals * 2, n_modes), dtype=np.float32)

        for i in range(self.n_petals):
            theta_rad = np.radians(self.angle_offset_deg + i * (360.0 / self.n_petals))
            nx, ny = -np.sin(theta_rad), np.cos(theta_rad)
            D = X * nx + Y * ny

            margin = self.spider_widths[i] * 1.5

            left_mask = (D < 0) & (D > -margin) & mask_amp
            right_mask = (D >= 0) & (D < margin) & mask_amp

            left_1d = left_mask[mask_amp]
            right_1d = right_mask[mask_amp]

            R_valid = R[left_mask | right_mask]
            if len(R_valid) == 0:
                continue

            R_in = np.min(R_valid)
            R_out = np.max(R_valid)

            R_1d = R[mask_amp]
            R_L = R_1d[left_1d]
            R_R = R_1d[right_1d]

            for m in range(n_modes):
                mode_L = B[m, left_1d]
                mode_R = B[m, right_1d]

                # Robust 1D Radial Regression LEFT
                if len(R_L) > 0:
                    mean_R_L = np.mean(R_L)
                    m0_L = np.mean(mode_L)
                    var_R_L = np.sum((R_L - mean_R_L)**2)
                    mT_L = np.sum((R_L - mean_R_L) * (mode_L - m0_L)) / var_R_L if var_R_L > 1e-6 else 0.0
                else:
                    mean_R_L, m0_L, mT_L = 0.0, 0.0, 0.0

                # Robust 1D Radial Regression RIGHT
                if len(R_R) > 0:
                    mean_R_R = np.mean(R_R)
                    m0_R = np.mean(mode_R)
                    var_R_R = np.sum((R_R - mean_R_R)**2)
                    mT_R = np.sum((R_R - mean_R_R) * (mode_R - m0_R)) / var_R_R if var_R_R > 1e-6 else 0.0
                else:
                    mean_R_R, m0_R, mT_R = 0.0, 0.0, 0.0

                h_in = (m0_R + mT_R * (R_in - mean_R_R)) - (m0_L + mT_L * (R_in - mean_R_L))
                h_out = (m0_R + mT_R * (R_out - mean_R_R)) - (m0_L + mT_L * (R_out - mean_R_L))

                H_full_cpu[2*i, m] = h_in
                H_full_cpu[2*i + 1, m] = h_out

        if np.sum(np.abs(H_full_cpu)) == 0.0:
             self.logger.error("CRITICAL ERROR: H_full is completely ZERO! Check your angle_offset_deg and n_petals!")
        else:
             self.logger.info("H_full matrix generated successfully.")

        self.H_full = self.to_xp(H_full_cpu, dtype=self.dtype)

        # 4. MAP GAPS BACK TO IDEAL PETALS
        H_petals_cpu = H_full_cpu @ C_petals_cpu # Shape: (12, 18)
        self.H_petals_dagger = self.to_xp(np.linalg.pinv(H_petals_cpu, rcond=self.rcond), dtype=self.dtype)

        self.logger.info(f"Unwrapper matrices ready. Piston Only Actuation: {self.piston_only}")

    def trigger_code(self):

        in_comm = self.local_inputs['in_comm'].value

        out_comm_val = in_comm.copy()
        out_ost_val = self.xp.zeros_like(in_comm)

        do_check = False
        if self.current_time >= self.start_time_t:
            if self.interval_time_t <= 0:
                do_check = True
            elif (self.current_time - self.last_correction_time) >= self.interval_time_t - 1e-6:
                do_check = True

        if do_check:
            # 1. Measure the exact physical gaps
            h_gaps = self.H_full @ in_comm

            # 2. Hard Limiter Logic
            h_err = self.xp.where(self.xp.abs(h_gaps) > self.thresh_in_nm, h_gaps, 0.0)

            if self.xp.any(h_err):
                # 3. Find the ideal petal amplitudes (18 DOFs) that explain the error
                delta_p = self.H_petals_dagger @ h_err

                # 4. SURGICAL FILTERING: Kill Tip/Tilt if requested
                if self.piston_only:
                    delta_p *= self.piston_mask

                # 5. Map the filtered petals back into the command basis
                delta_comm = self.C_petals @ delta_p

                # Apply corrections
                out_comm_val -= delta_comm
                out_ost_val = delta_comm

            self.last_correction_time = self.current_time

        self.outputs['out_comm'].set_value(out_comm_val)
        self.outputs['out_ost'].set_value(out_ost_val)

    def post_trigger(self):
        super().post_trigger()
        self.outputs['out_comm'].generation_time = self.current_time
        self.outputs['out_ost'].generation_time = self.current_time
