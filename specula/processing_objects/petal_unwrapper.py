from specula import np, cpuArray
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.ifunc import IFunc
from specula.data_objects.pupilstop import Pupilstop

class PetalUnwrapper(BaseProcessingObj):
    """
    Topological Unwrapper for Segmented Mirrors.
    Inspects the command vector to identify and eliminate Phase Wrapping errors
    (Island Effect) on the spiders, using integrals over the physical gaps.
    """
    def __init__(self,
                 ifunc: IFunc,
                 pupilstop: Pupilstop,
                 n_petals: int = 6,
                 angle_offset_deg: float = 90.0,
                 spider_widths: list = None,
                 thresh_in_nm: float = 350.0,
                 lambda_wfs: float = 700.0,    # <--- Aggiunto! Serve per la quantizzazione
                 n_petal_modes: int = 10,
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        
        self.n_petals = n_petals
        self.angle_offset_deg = angle_offset_deg
        self.thresh_in_nm = thresh_in_nm
        self.lambda_wfs = lambda_wfs
        self.n_petal_modes = n_petal_modes
        
        if spider_widths is None:
            self.spider_widths = [0.51] * self.n_petals
        else:
            self.spider_widths = spider_widths
            
        self.inputs['in_comm'] = InputValue(type=BaseValue)
        
        # out_comm is the cleaned command to send to the DM
        self.outputs['out_comm'] = BaseValue(target_device_idx=self.target_device_idx, precision=self.precision)
        # out_ost is the correction vector to feed back to the IIR filter's in_ost
        self.outputs['out_ost'] = BaseValue(target_device_idx=self.target_device_idx, precision=self.precision)

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
        """Pre-calculates the Observation Matrix H and its pseudo-inverse H_dagger offline."""
        self.logger.info("Initializing Topologic Unwrapper matrices...")
        
        # Forziamo i dati sulla CPU per l'inizializzazione geometrica
        mask_amp = cpuArray(pupilstop.A) > 0
        dim = mask_amp.shape[0]
        pitch = pupilstop.pixel_pitch
        
        # Create physical coordinates centered on pupil
        x_1d = (np.arange(dim) - dim / 2.0 + 0.5) * pitch
        X, Y = np.meshgrid(x_1d, x_1d)
        R = np.sqrt(X**2 + Y**2)
        
        # Estraiamo i modi e portiamoli su CPU per evitare crash con indici booleani numpy
        B_petals = cpuArray(ifunc.influence_function[-self.n_petal_modes:, :])
        
        # H matrix maps 10 modes to 12 physical heights (2 per spider)
        H_cpu = np.zeros((self.n_petals * 2, self.n_petal_modes))
        
        for i in range(self.n_petals):
            theta_deg = self.angle_offset_deg + i * (360.0 / self.n_petals)
            theta_rad = np.radians(theta_deg)
            
            # Normal vector to the spider line
            nx, ny = -np.sin(theta_rad), np.cos(theta_rad)
            
            # Orthogonal distance from the spider line
            D = X * nx + Y * ny
            
            # La maschera si allarga leggermente oltre il ragno reale (margin)
            margin = self.spider_widths[i] * 1.5 
            
            left_mask = (D < 0) & (D > -margin) & mask_amp
            right_mask = (D >= 0) & (D < margin) & mask_amp
            
            # 1D masks to index into the IFunc array
            left_1d = left_mask[mask_amp]
            right_1d = right_mask[mask_amp]
            
            R_valid = R[left_mask | right_mask]
            
            # Controllo di sicurezza: se la maschera non "acchiappa" nessun pixel
            if len(R_valid) == 0:
                self.logger.warning(f"Spider {i}: Margin too small or masked entirely. Integrals will be zero.")
                continue

            R_in = np.min(R_valid)
            R_out = np.max(R_valid)
            R_mid = (R_in + R_out) / 2.0
            delta_R = (R_out - R_in) / 2.0
            
            R_1d = R[mask_amp]
            
            for m in range(self.n_petal_modes):
                mode = B_petals[m, :]
                
                # Moment 0: Mean Step (Protetto contro divisione per zero se i pixel sono 0)
                m0_L = np.mean(mode[left_1d]) if np.any(left_1d) else 0.0
                m0_R = np.mean(mode[right_1d]) if np.any(right_1d) else 0.0
                S_mean = m0_R - m0_L
                
                # Moment 1: Differential Tilt (Twist)
                m1_L = np.mean((R_1d[left_1d] - R_mid) * mode[left_1d]) if np.any(left_1d) else 0.0
                m1_R = np.mean((R_1d[right_1d] - R_mid) * mode[right_1d]) if np.any(right_1d) else 0.0
                S_tilt = m1_R - m1_L
                
                # Extrapolate heights at physical borders (R_in and R_out)
                h_in = S_mean - delta_R * S_tilt
                h_out = S_mean + delta_R * S_tilt
                
                H_cpu[2*i, m] = h_in
                H_cpu[2*i + 1, m] = h_out
                
        # Send matrices to target device (CPU/GPU) per l'uso ad alta frequenza
        self.H = self.to_xp(H_cpu, dtype=self.dtype)
        # Standard Moore-Penrose pseudo-inverse (va benissimo perché lo spazio è isolato)
        self.H_dagger = self.xp.linalg.pinv(self.H)
        
        self.logger.info("Unwrapper matrices ready.")

    def trigger_code(self):
        in_comm = self.local_inputs['in_comm'].value
        
        # 1. Isolate the appended petal modes
        m_pet = in_comm[-self.n_petal_modes:]
        
        # 2. Virtual Topological Measurement (12 points)
        h_gaps = self.H @ m_pet
        
        # 3. Non-Linear Thresholding (Identify wrapping errors)
        h_err = self.xp.where(self.xp.abs(h_gaps) > self.thresh_in_nm,
                              self.xp.round(h_gaps / self.lambda_wfs) * self.lambda_wfs,
                              0.0)
        
        # Output buffers
        out_comm_val = in_comm.copy()
        out_ost_val = self.xp.zeros_like(in_comm)
        
        # 4. Correct if Island Effect is detected
        if self.xp.any(h_err):
            # Distribute error natively over the 10 appended DOFs
            delta_m = self.H_dagger @ h_err
            
            # Clean the DM command for this frame
            out_comm_val[-self.n_petal_modes:] -= delta_m
            
            # Create the reset vector for the IIR Filter (applies to next frame)
            out_ost_val[-self.n_petal_modes:] = delta_m
            
        self.outputs['out_comm'].set_value(out_comm_val)
        self.outputs['out_ost'].set_value(out_ost_val)