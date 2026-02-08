"""
Interaction Matrix Generator for Shack-Hartmann WFS using SynIM.

This processing object computes a full interaction matrix given mis-registration
parameters. It can be used standalone or connected to SPRINT estimator output
to generate the corrected IM.
"""

import synim.synim as synim
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.intmat import Intmat
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.source import Source
from specula.base_value import BaseValue
from specula.processing_objects.dm import DM
from specula.processing_objects.sh import SH
from specula.processing_objects.sh_slopec import ShSlopec
from specula import cpuArray, np


class ImShSynimGenerator(BaseProcessingObj):
    """
    Interaction Matrix Generator for Shack-Hartmann WFS using SynIM.
    
    Computes a full interaction matrix with specified mis-registration parameters.
    Can be connected to SPRINT estimator to use estimated parameters, or used
    standalone with fixed parameters.
    
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
    modes_index : list or None
        List of mode indices to compute (None = all modes)
    apply_absolute_slopes : bool
        Use absolute value of slopes (default: False)
    data_dir : str or None
        Directory to save IM (default: simul_params.root_dir)
    im_tag : str or None
        Tag for IM filename (default: 'im_sh_synim')
    overwrite : bool
        Overwrite existing IM file (default: False)
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
    
    Examples
    --------
    # Standalone usage with fixed parameters
    >>> im_gen = ImShSynimGenerator(
    ...     simul_params=simul_params,
    ...     dm=dm,
    ...     slopec=slopec,
    ...     source=source,
    ...     wfs=wfs
    ... )
    >>> im_gen.setup()
    >>> im = im_gen.generate_im([2.0, 1.5, 1.0, 0.02])  # shift_x, shift_y, rot, mag
    
    # Connected to SPRINT
    >>> sprint = SprintShSynim(...)
    >>> im_gen = ImShSynimGenerator(...)
    >>> im_gen.inputs['in_misreg_params'].set(sprint.outputs['out_misreg_params'])
    >>> im_gen.setup()
    >>> # IM is automatically updated when SPRINT estimates new parameters
    """

    def __init__(self,
                 simul_params: SimulParams,
                 dm: DM,
                 slopec: ShSlopec,
                 source: Source,
                 wfs: SH,
                 modes_index: list = None,
                 apply_absolute_slopes: bool = False,
                 data_dir: str = None,
                 im_tag: str = None,
                 overwrite: bool = False,
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

        # Mode configuration
        self.modes_index = modes_index
        self.apply_absolute_slopes = apply_absolute_slopes

        # File I/O
        self.data_dir = data_dir or simul_params.root_dir
        self.im_tag = im_tag or 'im_sh_synim'
        self.overwrite = overwrite

        # Pupil parameters
        self.pup_diam_m = simul_params.pixel_pupil * simul_params.pixel_pitch
        self.pup_mask = None
        self.ifunc_3d = None
        self.idx_valid_sa = None

        # Create output
        self.output_intmat = Intmat(
            nmodes=0,  # Set in setup
            nslopes=0,  # Set in setup
            target_device_idx=target_device_idx,
            precision=precision
        )

        # Setup connections
        self.inputs['in_misreg_params'] = InputValue(type=BaseValue, optional=True)
        self.outputs['out_intmat'] = self.output_intmat

    def setup(self):
        """Initialize and extract parameters"""
        super().setup()

        # Extract DM parameters
        ifunc_3d_full = cpuArray(self.dm.ifunc_obj.ifunc_2d_to_3d(normalize=True))

        if self.modes_index is not None:
            self.ifunc_3d = ifunc_3d_full[:, :, self.modes_index]
            nmodes = len(self.modes_index)
        else:
            self.ifunc_3d = ifunc_3d_full
            nmodes = ifunc_3d_full.shape[2]
            self.modes_index = list(range(nmodes))

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

        if self.verbose:
            print(f"\n{self.__class__.__name__} initialized:")
            print(f"  WFS type: Shack-Hartmann (SynIM backend)")
            print(f"  Subapertures: {self.wfs.subap_on_diameter}x{self.wfs.subap_on_diameter}")
            print(f"  Valid subapertures: {len(self.idx_valid_sa)}")
            print(f"  Number of modes: {nmodes}")
            print(f"  Number of slopes: {nslopes}")
            print(f"  FOV: {self.wfs.subap_wanted_fov:.2f} arcsec")

    def trigger_code(self):
        """Generate IM when input changes or on demand"""
        t = self.current_time

        # Get mis-registration parameters
        in_misreg = self.local_inputs.get('in_misreg_params')

        if in_misreg is not None:
            misreg_params = cpuArray(in_misreg.value)
        else:
            # Default: perfect registration
            misreg_params = np.zeros(4)

        if self.verbose:
            print(f"\nGenerating IM with mis-registration:")
            print(f"  shift_x: {misreg_params[0]:.3f} px")
            print(f"  shift_y: {misreg_params[1]:.3f} px")
            print(f"  rotation: {misreg_params[2]:.3f} deg")
            print(f"  magnification: {misreg_params[3]:.6f}")

        # Generate IM
        im = self.generate_im(misreg_params)

        # Update output
        self.output_intmat.intmat = self.to_xp(im, dtype=self.dtype)
        self.output_intmat.generation_time = t

    def generate_im(self, misreg_params):
        """
        Generate interaction matrix with given mis-registration.
        
        Parameters
        ----------
        misreg_params : array_like, shape (4,) or (6,)
            Mis-registration parameters:
            [shift_x, shift_y, rotation, magnification(, magn_x, magn_y)]
        
        Returns
        -------
        im : ndarray, shape (nslopes, nmodes)
            Interaction matrix
        """
        misreg_params = np.asarray(misreg_params)

        # Extract parameters
        shift_x = float(misreg_params[0])
        shift_y = float(misreg_params[1])
        rotation = float(misreg_params[2])
        magnification = 1.0 + float(misreg_params[3])

        # Get source parameters
        gs_pol_coo = tuple(cpuArray(self.source.polar_coordinates))
        gs_height = self.source.height if self.source.height != float('inf') else float('inf')

        # Compute IM with SynIM
        im = synim.interaction_matrix(
            pup_diam_m=self.pup_diam_m,
            pup_mask=self.pup_mask,
            dm_array=self.ifunc_3d,
            dm_mask=self.dm.mask.T if hasattr(self.dm.mask, 'T') else cpuArray(self.dm.mask).T,
            dm_height=0.0,
            dm_rotation=0.0,
            gs_pol_coo=gs_pol_coo,
            gs_height=gs_height,
            wfs_nsubaps=self.wfs.subap_on_diameter,
            wfs_rotation=rotation,
            wfs_translation=(shift_x, shift_y),
            wfs_mag_global=magnification,
            wfs_fov_arcsec=self.wfs.subap_wanted_fov,
            idx_valid_sa=self.idx_valid_sa,
            verbose=False,
            specula_convention=True
        )

        if self.apply_absolute_slopes:
            im = np.abs(im)

        return im
