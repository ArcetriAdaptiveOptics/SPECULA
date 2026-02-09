"""
Interaction Matrix Generator for Shack-Hartmann WFS using SynIM.

This processing object computes a full interaction matrix given mis-registration
parameters, and optionally computes the corresponding reconstruction matrix.
Can be connected to SPRINT estimator output to generate corrected IM and RM.
"""

from specula.lib.synim_utils import compute_im_synim
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.source import Source
from specula.data_objects.ifunc import IFunc
from specula.data_objects.m2c import M2C
from specula.base_value import BaseValue
from specula.processing_objects.dm import DM
from specula.processing_objects.sh import SH
from specula.processing_objects.sh_slopec import ShSlopec
from specula import cpuArray, np, xp
from typing import Union


class ImShSynimGenerator(BaseProcessingObj):
    """
    Interaction Matrix and Reconstruction Matrix Generator for Shack-Hartmann WFS.
    
    Computes interaction matrix and reconstruction matrix with specified 
    mis-registration parameters using SynIM geometric model.
    
    Can be connected to SPRINT estimator to automatically generate corrected
    IM and RM when new mis-registration parameters are estimated.
    
    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters
    dm : DM
        Deformable mirror object
    slopec : ShSlopec
        Shack-Hartmann slope computer
    source : Source
        Guide star source
    wfs : SH
        Shack-Hartmann WFS object
    compute_rec : bool
        Compute reconstruction matrix (default: True)
    rec_nmodes : int or None
        Number of modes for reconstruction (None = same as IM)
    mmse : bool
        Use MMSE reconstruction instead of pseudo-inverse (default: False)
    r0 : float
        Fried parameter for MMSE [m] (default: 0.15)
    L0 : float
        Outer scale for MMSE [m] (default: 25.0)
    noise_cov : float, ndarray, list, or None
        Noise covariance for MMSE (required if mmse=True)
    target_device_idx : int or None
        GPU device index
    precision : int or None
        Numerical precision
    
    Inputs
    ------
    in_misreg_params : BaseValue, optional
        Mis-registration parameters [shift_x, shift_y, rotation, magnification]
        If not connected, uses zeros (perfect registration)
    
    Outputs
    -------
    out_intmat : Intmat
        Generated interaction matrix
    out_recmat : Recmat
        Generated reconstruction matrix (if compute_rec=True)
    
    Examples
    --------
    # Basic usage with pseudo-inverse
    >>> im_gen = ImShSynimGenerator(
    ...     simul_params=simul_params,
    ...     dm=dm,
    ...     slopec=slopec,
    ...     source=source,
    ...     wfs=wfs,
    ...     compute_rec=True
    ... )
    
    # With MMSE reconstruction
    >>> im_gen = ImShSynimGenerator(
    ...     simul_params=simul_params,
    ...     dm=dm,
    ...     slopec=slopec,
    ...     source=source,
    ...     wfs=wfs,
    ...     compute_rec=True,
    ...     mmse=True,
    ...     r0=0.15,
    ...     L0=25.0,
    ...     noise_cov=0.1
    ... )
    
    # Connected to SPRINT
    >>> sprint = SprintShSynim(...)
    >>> im_gen = ImShSynimGenerator(...)
    >>> im_gen.inputs['in_misreg_params'].set(sprint.outputs['out_misreg_params'])
    """

    def __init__(self,
                 simul_params: SimulParams,
                 dm: DM,
                 slopec: ShSlopec,
                 source: Source,
                 wfs: SH,
                 compute_rec: bool = True,
                 rec_nmodes: int = None,
                 mmse: bool = False,
                 r0: float = 0.15,
                 L0: float = 25.0,
                 noise_cov: Union[float, np.ndarray, list] = None,
                 verbose: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Validate WFS type
        if not isinstance(wfs, SH):
            raise ValueError(f"ImShSynimGenerator requires SH WFS, got {type(wfs).__name__}")
        if not isinstance(slopec, ShSlopec):
            raise ValueError(f"ImShSynimGenerator requires ShSlopec, got {type(slopec).__name__}")

        # Store references
        self.simul_params = simul_params
        self.dm = dm
        self.slopec = slopec
        self.source = source
        self.wfs = wfs

        # Reconstruction configuration
        self.compute_rec = compute_rec
        self.rec_nmodes = rec_nmodes
        self.mmse = mmse
        self.r0 = r0
        self.L0 = L0

        # Validate MMSE parameters
        if mmse and noise_cov is None:
            raise ValueError('noise_cov must be provided for MMSE reconstruction')

        if noise_cov is None:
            self.noise_cov = None
        elif isinstance(noise_cov, list):
            self.noise_cov = [self.to_xp(nc) for nc in noise_cov]
        else:
            self.noise_cov = self.to_xp(noise_cov)

        self.verbose = verbose

        # Pupil parameters
        self.pup_diam_m = simul_params.pixel_pupil * simul_params.pixel_pitch
        self.pup_mask = None
        self.ifunc_3d = None
        self.idx_valid_sa = None

        # Create outputs
        self.output_intmat = Intmat(
            nmodes=0,  # Set in setup
            nslopes=0,  # Set in setup
            target_device_idx=target_device_idx,
            precision=precision
        )

        self.output_recmat = None
        if compute_rec:
            # Create empty recmat, will be sized in setup
            self.output_recmat = Recmat(
                recmat=self.xp.zeros((1, 1), dtype=self.dtype),
                target_device_idx=target_device_idx,
                precision=precision
            )

        # Setup connections
        self.inputs['in_misreg_params'] = InputValue(type=BaseValue, optional=True)
        self.outputs['out_intmat'] = self.output_intmat
        if compute_rec:
            self.outputs['out_recmat'] = self.output_recmat

    def setup(self):
        """Initialize and extract parameters"""
        super().setup()

        # Extract DM parameters
        ifunc_3d_full = cpuArray(self.dm.ifunc_obj.ifunc_2d_to_3d(normalize=True))
        self.ifunc_3d = ifunc_3d_full
        nmodes = ifunc_3d_full.shape[2]

        self.pup_mask = cpuArray(self.dm.mask)

        # Extract valid subapertures
        subapdata = self.slopec.subapdata
        display_map = cpuArray(subapdata.display_map)
        nx = subapdata.nx
        idx_i = display_map // nx
        idx_j = display_map % nx
        self.idx_valid_sa = np.column_stack((idx_i, idx_j))

        # Initialize output IM size
        nslopes = len(subapdata.display_map) * 2  # x and y slopes
        self.output_intmat.set_nmodes(nmodes)
        self.output_intmat.set_nslopes(nslopes)

        # Set rec_nmodes default
        if self.rec_nmodes is None:
            self.rec_nmodes = nmodes

        # Initialize recmat size if needed
        if self.compute_rec:
            recmat_shape = (self.rec_nmodes, nslopes)
            self.output_recmat.recmat = self.xp.zeros(recmat_shape, dtype=self.dtype)

        if self.verbose: # pragma: no cover
            print(f"\n{self.__class__.__name__} initialized:")
            print(f"  WFS type: Shack-Hartmann (SynIM backend)")
            print(f"  Subapertures: {self.wfs.subap_on_diameter}x{self.wfs.subap_on_diameter}")
            print(f"  Valid subapertures: {len(self.idx_valid_sa)}")
            print(f"  Number of IM modes: {nmodes}")
            print(f"  Number of slopes: {nslopes}")
            print(f"  FOV: {self.wfs.subap_wanted_fov:.2f} arcsec")
            if self.compute_rec:
                print(f"  Compute reconstruction: Yes")
                print(f"  Number of REC modes: {self.rec_nmodes}")
                print(f"  Reconstruction method: {'MMSE' if self.mmse else 'Pseudo-inverse'}")

    def trigger_code(self):
        """Generate IM and optionally REC when input changes or on demand"""
        t = self.current_time

        # Get mis-registration parameters
        in_misreg = self.local_inputs.get('in_misreg_params')

        if in_misreg is not None:
            misreg_params = cpuArray(in_misreg.value)
        else:
            # Default: perfect registration
            misreg_params = np.zeros(4)

        if self.verbose: # pragma: no cover
            print(f"\nGenerating IM with mis-registration:")
            print(f"  shift_x: {misreg_params[0]:.3f} px")
            print(f"  shift_y: {misreg_params[1]:.3f} px")
            print(f"  rotation: {misreg_params[2]:.3f} deg")
            print(f"  magnification: {misreg_params[3]:.6f}")

        # Generate IM
        im = self.generate_im(misreg_params)

        # Update output IM
        self.output_intmat.intmat = self.to_xp(im, dtype=self.dtype)
        self.output_intmat.generation_time = t

        # Generate REC if requested
        if self.compute_rec:
            if self.verbose: # pragma: no cover
                print(f"  Computing reconstruction matrix...")

            rec = self.generate_rec()

            # Update output REC
            self.output_recmat.set_value(rec.recmat)
            self.output_recmat.generation_time = t

            if self.verbose: # pragma: no cover
                print(f"  REC matrix shape: {rec.recmat.shape}")

    def generate_im(self, misreg_params):
        """
        Generate interaction matrix with given mis-registration.
        
        Parameters
        ----------
        misreg_params : array_like
            Mis-registration parameters:
            - If length 4: [shift_x, shift_y, rotation, magnification]
            - If length 6: [shift_x, shift_y, rotation, mag_global,
                            anamorphosis_90, anamorphosis_45]
        
        Returns
        -------
        im : ndarray, shape (nslopes, nmodes)
            Interaction matrix
        """
        return compute_im_synim(
            misreg_params=misreg_params,
            pup_diam_m=self.pup_diam_m,
            pup_mask=self.pup_mask,
            ifunc_3d=self.ifunc_3d,
            dm_mask=self.dm.mask,
            source_polar_coords=self.source.polar_coordinates,
            source_height=self.source.height,
            wfs_nsubaps=self.wfs.subap_on_diameter,
            wfs_fov_arcsec=self.wfs.subap_wanted_fov,
            idx_valid_sa=self.idx_valid_sa,
            apply_absolute_slopes=False,
            verbose=self.verbose
        )

    def generate_rec(self):
        """
        Generate reconstruction matrix from current interaction matrix.
        
        Returns
        -------
        rec : Recmat
            Reconstruction matrix
        """
        if self.mmse:
            # MMSE reconstruction (same as RecCalibrator)
            diameter = self.pup_diam_m
            modal_base = IFunc(
                ifunc=self.dm.ifunc_obj.ifunc,
                mask=self.dm.mask,
                target_device_idx=self.target_device_idx,
                precision=self.precision
            )

            if self.dm.m2c is not None:
                m2c = M2C(
                    self.dm.m2c,
                    target_device_idx=self.target_device_idx,
                    precision=self.precision
                )
            else:
                m2c = None

            rec = self.output_intmat.generate_rec_mmse(
                self.r0, self.L0, diameter, modal_base,
                self.noise_cov, nmodes=self.rec_nmodes, m2c=m2c
            )
        else:
            # Simple pseudo-inverse
            rec = self.output_intmat.generate_rec(self.rec_nmodes)

        return rec
