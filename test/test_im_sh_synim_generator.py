import unittest

import specula
specula.init(0)  # Default target device

from specula.data_objects.simul_params import SimulParams
from specula.data_objects.source import Source
from specula.data_objects.ifunc import IFunc
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.subap_data import SubapData
from specula.base_value import BaseValue
from specula.processing_objects.dm import DM
from specula.processing_objects.sh import SH
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.im_sh_synim_generator import ImShSynimGenerator
import synim.synim as synim
from specula import np, cpuArray

from test.specula_testlib import cpu_and_gpu


def create_test_system():
    """Create a minimal SCAO system for testing IM generator"""

    # Simulation parameters
    simul_params = SimulParams(
        time_step=1e-3,  # 1ms
        pixel_pupil=100,
        pixel_pitch=8.0/100  # 8m telescope
    )

    # Source (on-axis NGS)
    source = Source(
        polar_coordinates=[0.0, 0.0],  # On-axis
        magnitude=8.0,
        wavelengthInNm=500.0
    )

    # DM - Zernike modes
    n_modes = 50  # Smaller for faster tests
    ifunc = IFunc(
        type_str='zernike',
        npixels=simul_params.pixel_pupil,
        nmodes=n_modes,
        obsratio=0.0
    )

    dm = DM(
        simul_params=simul_params,
        ifunc=ifunc,
        height=0.0
    )

    # Create SubapData (valid subapertures)
    subap_on_diameter = 10
    subap_npx = 10
    n_subap = subap_on_diameter

    mask_subap = np.ones((subap_on_diameter*subap_npx, subap_on_diameter*subap_npx))
    idxs = {}
    map_dict = {}

    count = 0
    for i in range(subap_on_diameter):
        for j in range(subap_on_diameter):
            mask_subap *= 0
            mask_subap[i*subap_npx:(i+1)*subap_npx, j*subap_npx:(j+1)*subap_npx] = 1
            idxs[count] = np.where(mask_subap == 1)
            map_dict[count] = j * subap_on_diameter + i
            count += 1

    v = np.zeros((len(idxs), subap_npx*subap_npx), dtype=int)
    m = np.zeros(len(idxs), dtype=int)
    for k, idx in idxs.items():
        v[k] = np.ravel_multi_index(idx, mask_subap.shape)
        m[k] = map_dict[k]

    subapdata = SubapData(
        idxs=v,
        display_map=m,
        nx=n_subap,
        ny=n_subap
    )

    # SH WFS
    pxscale_arcsec = 0.5
    wfs = SH(
        wavelengthInNm=500,
        subap_wanted_fov=subap_npx * pxscale_arcsec,
        sensor_pxscale=pxscale_arcsec,
        subap_on_diameter=subap_on_diameter,
        subap_npx=subap_npx
    )

    # Create input electric field for WFS setup
    ef = ElectricField(
        simul_params.pixel_pupil,
        simul_params.pixel_pupil,
        simul_params.pixel_pitch,
        S0=1
    )
    ef.generation_time = 1

    # Connect and setup WFS
    wfs.inputs['in_ef'].set(ef)
    wfs.setup()

    # Slopec
    slopec = ShSlopec(
        subapdata=subapdata
    )

    return simul_params, source, dm, wfs, slopec


def generate_reference_im_synim(simul_params, source, dm, wfs, slopec,
                                shift_x=0.0, shift_y=0.0, rotation=0.0,
                                magnification=0.0, xp=np):
    """Generate reference IM using SynIM directly"""

    # Extract parameters
    pup_diam_m = simul_params.pixel_pupil * simul_params.pixel_pitch
    pup_mask = dm.mask

    # Get 3D influence functions
    ifunc_3d = dm.ifunc_obj.ifunc_2d_to_3d(normalize=True)

    # Get valid subapertures from slopec
    subapdata = slopec.subapdata
    display_map = subapdata.display_map
    nx = subapdata.nx
    idx_i = display_map // nx
    idx_j = display_map % nx
    idx_valid_sa = xp.column_stack((idx_i, idx_j))

    # Source coordinates
    gs_pol_coo = tuple(cpuArray(source.polar_coordinates))
    gs_height = source.height if source.height != float('inf') else float('inf')

    # WFS parameters
    subap_on_diameter = wfs.subap_on_diameter
    subap_fov = wfs.subap_wanted_fov

    # Compute IM with SynIM
    im_ref = synim.interaction_matrix(
        pup_diam_m=pup_diam_m,
        pup_mask=pup_mask,
        dm_array=ifunc_3d,
        dm_mask=pup_mask.T,
        dm_height=0.0,
        dm_rotation=0.0,
        gs_pol_coo=gs_pol_coo,
        gs_height=gs_height,
        wfs_nsubaps=subap_on_diameter,
        wfs_rotation=float(rotation),
        wfs_translation=(float(shift_x), float(shift_y)),
        wfs_mag_global=1.0 + float(magnification),
        wfs_fov_arcsec=subap_fov,
        idx_valid_sa=idx_valid_sa,
        verbose=False,
        specula_convention=True
    )

    return im_ref


class TestImShSynimGenerator(unittest.TestCase):

    verbose = False  # Set to True to print debug info during tests

    @cpu_and_gpu
    def test_im_generator_no_misreg(self, target_device_idx, xp):
        """Test IM generation with no mis-registration (identity)"""

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing ImShSynimGenerator with perfect registration")
            print(f"  target_device={target_device_idx}")
            print(f"{'='*70}")

        # Create test system
        simul_params, source, dm, wfs, slopec = create_test_system()

        # Create IM generator
        im_gen = ImShSynimGenerator(
            simul_params=simul_params,
            dm=dm,
            slopec=slopec,
            source=source,
            wfs=wfs,
            compute_rec=False,  # Don't compute REC for this test
            target_device_idx=target_device_idx,
            precision=1
        )

        # Setup (no input connected, will use default params)
        im_gen.setup()

        # Trigger IM generation
        im_gen.trigger_code()

        # Get generated IM
        im_generated = specula.cpuArray(im_gen.output_intmat.intmat)

        if self.verbose: # pragma: no cover
            print(f"\nGenerated IM shape: {im_generated.shape}")
            print(f"Generated IM RMS: {np.sqrt(np.mean(im_generated**2)):.3e}")

        # Generate reference IM with SynIM directly
        im_ref = generate_reference_im_synim(simul_params, source, dm, wfs, slopec,
                                            shift_x=0.0, shift_y=0.0,
                                            rotation=0.0, magnification=0.0, xp=xp)

        if self.verbose: # pragma: no cover
            print(f"Reference IM shape: {im_ref.shape}")
            print(f"Reference IM RMS: {np.sqrt(np.mean(im_ref**2)):.3e}")

        # Compare
        im_diff = im_generated - im_ref
        rms_diff = np.sqrt(np.mean(im_diff**2))
        rel_diff = rms_diff / np.sqrt(np.mean(im_ref**2))

        if self.verbose: # pragma: no cover
            print(f"\nDifference RMS: {rms_diff:.3e}")
            print(f"Relative difference: {rel_diff*100:.3f}%")

        # Should be essentially identical
        self.assertLess(rel_diff, 1e-10, "Generated IM should match reference")

    @cpu_and_gpu
    def test_im_generator_with_misreg(self, target_device_idx, xp):
        """Test IM generation with mis-registration"""

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing ImShSynimGenerator with mis-registration")
            print(f"  target_device={target_device_idx}")
            print(f"{'='*70}")

        # Define mis-registration parameters
        shift_x = 2.0
        shift_y = -1.5
        rotation = 1.5
        magnification = 0.03

        if self.verbose: # pragma: no cover
            print(f"\nMis-registration parameters:")
            print(f"  shift_x: {shift_x} px")
            print(f"  shift_y: {shift_y} px")
            print(f"  rotation: {rotation} deg")
            print(f"  magnification: {magnification}")

        # Create test system
        simul_params, source, dm, wfs, slopec = create_test_system()

        # Create BaseValue with mis-registration parameters
        misreg_params = BaseValue(
            value=np.array([shift_x, shift_y, rotation, magnification]),
            target_device_idx=target_device_idx,
            precision=1
        )

        # Create IM generator
        im_gen = ImShSynimGenerator(
            simul_params=simul_params,
            dm=dm,
            slopec=slopec,
            source=source,
            wfs=wfs,
            compute_rec=False,
            target_device_idx=target_device_idx,
            precision=1
        )

        # Connect mis-registration input
        im_gen.inputs['in_misreg_params'].set(misreg_params)

        # Setup
        im_gen.setup()

        # Trigger IM generation
        im_gen.trigger_code()

        # Get generated IM
        im_generated = specula.cpuArray(im_gen.output_intmat.intmat)

        # Generate reference IM with SynIM directly
        im_ref = generate_reference_im_synim(simul_params, source, dm, wfs, slopec,
                                            shift_x, shift_y, rotation, magnification,
                                            xp=xp)

        # Compare
        im_diff = im_generated - im_ref
        rms_diff = np.sqrt(np.mean(im_diff**2))
        rel_diff = rms_diff / np.sqrt(np.mean(im_ref**2))

        if self.verbose: # pragma: no cover
            print(f"\nGenerated IM RMS: {np.sqrt(np.mean(im_generated**2)):.3e}")
            print(f"Reference IM RMS: {np.sqrt(np.mean(im_ref**2)):.3e}")
            print(f"Difference RMS: {rms_diff:.3e}")
            print(f"Relative difference: {rel_diff*100:.3f}%")

        # Should match
        self.assertLess(rel_diff, 1e-7,
                        "Generated IM should match reference with mis-registration")

    @cpu_and_gpu
    def test_im_generator_with_rec(self, target_device_idx, xp):
        """Test IM and REC generation together"""

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing ImShSynimGenerator with REC computation")
            print(f"  target_device={target_device_idx}")
            print(f"{'='*70}")

        # Create test system
        simul_params, source, dm, wfs, slopec = create_test_system()

        # Create IM generator with REC
        rec_nmodes = 30  # Subset of modes
        im_gen = ImShSynimGenerator(
            simul_params=simul_params,
            dm=dm,
            slopec=slopec,
            source=source,
            wfs=wfs,
            compute_rec=True,  # Enable REC computation
            rec_nmodes=rec_nmodes,
            mmse=False,  # Use pseudo-inverse
            target_device_idx=target_device_idx,
            precision=1
        )

        # Setup
        im_gen.setup()

        # Trigger generation
        im_gen.trigger_code()

        # Get generated IM and REC
        im_generated = specula.cpuArray(im_gen.output_intmat.intmat)
        rec_generated = specula.cpuArray(im_gen.output_recmat.recmat)

        if self.verbose: # pragma: no cover
            print(f"\nGenerated IM shape: {im_generated.shape}")
            print(f"Generated REC shape: {rec_generated.shape}")

        # Check shapes
        nslopes, nmodes_im = im_generated.shape
        nmodes_rec, nslopes_rec = rec_generated.shape

        self.assertEqual(nslopes, nslopes_rec, "Slopes dimension should match")
        self.assertEqual(nmodes_rec, rec_nmodes, "REC should have requested number of modes")
        self.assertLessEqual(nmodes_rec, nmodes_im, "REC modes should be <= IM modes")

        # Test REC by applying to IM (should give identity for first rec_nmodes modes)
        # Using only first rec_nmodes columns of IM
        im_subset = im_generated[:, :rec_nmodes]
        identity_test = rec_generated @ im_subset

        if self.verbose: # pragma: no cover
            print(f"\nIdentity test (REC @ IM_subset):")
            print(f"  Expected: Identity matrix {rec_nmodes}x{rec_nmodes}")
            print(f"  Diagonal mean: {np.mean(np.diag(identity_test)):.3f}")
            print(f"  Off-diagonal RMS:"
                f" {np.sqrt(np.mean((identity_test - np.eye(rec_nmodes))**2)):.3e}")

        # Should be close to identity
        identity_error = np.sqrt(np.mean((identity_test - np.eye(rec_nmodes))**2))
        self.assertLess(identity_error, 1e-2, "REC @ IM should be close to identity")

    @cpu_and_gpu
    def test_im_generator_generate_im_method(self, target_device_idx, xp):
        """Test direct generate_im() method"""

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing ImShSynimGenerator.generate_im() method")
            print(f"  target_device={target_device_idx}")
            print(f"{'='*70}")

        # Create test system
        simul_params, source, dm, wfs, slopec = create_test_system()

        # Create IM generator
        im_gen = ImShSynimGenerator(
            simul_params=simul_params,
            dm=dm,
            slopec=slopec,
            source=source,
            wfs=wfs,
            compute_rec=False,
            target_device_idx=target_device_idx,
            precision=1
        )

        # Setup
        im_gen.setup()

        # Test with different mis-registration parameters
        test_cases = [
            ([0.0, 0.0, 0.0, 0.0], "No mis-registration"),
            ([2.0, 1.5, 1.0, 0.02], "Small mis-registration"),
            ([5.0, -3.0, 3.0, 0.05], "Large mis-registration"),
        ]

        for misreg_params, description in test_cases:
            if self.verbose: # pragma: no cover
                print(f"\nTest case: {description}")
                print(f"  Parameters: {misreg_params}")

            # Generate IM using method
            im_generated = im_gen.generate_im(misreg_params)

            # Generate reference
            im_ref = generate_reference_im_synim(
                simul_params, source, dm, wfs, slopec,
                misreg_params[0], misreg_params[1], misreg_params[2], misreg_params[3],
                xp=xp
            )

            # Compare
            im_diff = im_generated - im_ref
            rms_diff = np.sqrt(np.mean(im_diff**2))
            rel_diff = rms_diff / np.sqrt(np.mean(im_ref**2))

            if self.verbose: # pragma: no cover
                print(f"  Relative difference: {rel_diff*100:.3f}%")

            self.assertLess(rel_diff, 1e-10, f"generate_im()"
                            f" should match reference for {description}")
