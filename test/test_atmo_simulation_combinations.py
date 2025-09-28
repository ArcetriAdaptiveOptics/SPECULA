import unittest
import os
import shutil
import glob
import specula
specula.init(-1, precision=1)  # Default target device

from specula import np
from specula.simul import Simul
from astropy.io import fits

import tempfile
import yaml
import matplotlib.pyplot as plt

class TestAtmoSimulationCombinations(unittest.TestCase):
    """Test AtmoEvolution and AtmoInfiniteEvolution with different pupil sizes, pixel pitches, and L0 values"""

    def setUp(self):
        """Set up test environment"""
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.base_config = 'params_atmo_test.yml'
        
        # Get current working directory
        self.cwd = os.getcwd()
        
        # Define test combinations
        self.pixel_pupils = [128, 256, 512]
        self.pupil_diameters = [1.0, 8.0, 40.0]  # meters
        self.L0_values = [10.0, 25.0, 100.0]     # meters - outer scale
        
        # Calculate all combinations
        self.combinations = []
        for pixel_pupil in self.pixel_pupils:
            for diameter in self.pupil_diameters:
                for L0 in self.L0_values:
                    pixel_pitch = diameter / pixel_pupil
                    self.combinations.append({
                        'pixel_pupil': pixel_pupil,
                        'diameter': diameter,
                        'pixel_pitch': pixel_pitch,
                        'L0': L0,
                        'name': f'pp{pixel_pupil}_d{diameter:.0f}m_L0{L0:.0f}m'
                    })
        
        print(f"Testing {len(self.combinations)} combinations:")
        for combo in self.combinations:
            print(f"  {combo['name']}: pixel_pupil={combo['pixel_pupil']}, "
                  f"diameter={combo['diameter']:.1f}m, pixel_pitch={combo['pixel_pitch']:.6f}m, "
                  f"L0={combo['L0']:.1f}m")

    @classmethod
    def tearDownClass(cls):
        """Clean up after test by removing generated files"""
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        calibdir = os.path.join(os.path.dirname(__file__), 'calib')

        # Clean up phasescreens files
        phasescreen_files = glob.glob(os.path.join(calibdir, 'phasescreens', 'ps_seed*.fits'))
        for f in phasescreen_files:
            try:
                os.remove(f)
            except:
                pass
        
        # Remove test output directories
        output_dirs = glob.glob(os.path.join(datadir, '2*'))
        for output_dir in output_dirs:
            if os.path.isdir(output_dir):
                try:
                    shutil.rmtree(output_dir)
                except:
                    pass

    def create_override_config(self, combo):
        """Create temporary override configuration for a specific combination"""
        override_dict = {
            'main_override': {
                'pixel_pupil': combo['pixel_pupil'],
                'pixel_pitch': combo['pixel_pitch'],
                'total_time': 10.0  # 10 seconds simulation
            },
            'modal_analysis1_override': {
                'npixels': combo['pixel_pupil']
            },
            'modal_analysis2_override': {
                'npixels': combo['pixel_pupil']
            },
            'atmosphere_override': {
                'L0': [combo['L0']]  # L0 is typically a list for multi-layer atmosphere
            }
        }
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
        yaml.dump(override_dict, temp_file)
        temp_file.close()
        return temp_file.name

    def run_single_combination(self, combo):
        """Run simulation for a single parameter combination"""
        print(f"\n{'='*60}")
        print(f"Running combination: {combo['name']}")
        print(f"  pixel_pupil: {combo['pixel_pupil']}")
        print(f"  diameter: {combo['diameter']:.1f}m")
        print(f"  pixel_pitch: {combo['pixel_pitch']:.6f}m")
        print(f"  L0: {combo['L0']:.1f}m")
        print(f"{'='*60}")
        
        # Create override config
        override_path = self.create_override_config(combo)
        
        try:
            # Run simulation
            simul = Simul(self.base_config, override_path)
            simul.run()
            
            # Find output directory
            output_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
            self.assertTrue(output_dirs, f"No output directory found for {combo['name']}")
            latest_output_dir = output_dirs[-1]
            
            # Load results
            modes1_path = os.path.join(latest_output_dir, 'modes1.fits')
            modes2_path = os.path.join(latest_output_dir, 'modes2.fits')
            
            self.assertTrue(os.path.exists(modes1_path), 
                           f"modes1.fits not found for {combo['name']}")
            self.assertTrue(os.path.exists(modes2_path),
                           f"modes2.fits not found for {combo['name']}")

            # Read modal coefficients
            with fits.open(modes1_path) as hdul:
                modes1 = hdul[0].data
            with fits.open(modes2_path) as hdul:
                modes2 = hdul[0].data

            # Compute RMS
            rms_modes1 = np.sqrt(np.mean(modes1**2, axis=0))
            rms_modes2 = np.sqrt(np.mean(modes2**2, axis=0))
            
            print(f"Loaded data: modes1 shape={modes1.shape}, modes2 shape={modes2.shape}")
            print(f"RMS stats - modes1: min={np.min(rms_modes1):.3f}, max={np.max(rms_modes1):.3f}")
            print(f"RMS stats - modes2: min={np.min(rms_modes2):.3f}, max={np.max(rms_modes2):.3f}")
            
            return {
                'combo': combo,
                'rms_modes1': rms_modes1,
                'rms_modes2': rms_modes2,
                'modes1_shape': modes1.shape,
                'modes2_shape': modes2.shape
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(override_path):
                os.remove(override_path)

    def test_all_combinations(self):
        """Run simulations for all parameter combinations and display results"""
        
        # Change to test directory
        os.chdir(os.path.dirname(__file__))
        
        # Store results for all combinations
        all_results = []
        
        # Run each combination
        for i, combo in enumerate(self.combinations):
            print(f"\n{'-'*60}")
            print(f"Progress: {i+1}/{len(self.combinations)} combinations")
            
            try:
                result = self.run_single_combination(combo)
                all_results.append(result)
                print(f"✓ Completed: {combo['name']}")
                
            except Exception as e:
                print(f"✗ Failed: {combo['name']} - {str(e)}")
                continue
        
        # Display results
        self.display_results(all_results)
        
        # Basic assertions
        self.assertGreater(len(all_results), 0, "No simulations completed successfully")
        print(f"\n✓ Successfully completed {len(all_results)}/{len(self.combinations)} combinations")

    def display_results(self, all_results):
        """Display RMS comparison plots for all combinations"""
        if not all_results:
            print("No results to display")
            return
        
        # Group results by L0 values for separate plots
        results_by_L0 = {}
        for result in all_results:
            L0 = result['combo']['L0']
            if L0 not in results_by_L0:
                results_by_L0[L0] = []
            results_by_L0[L0].append(result)
        
        # Create separate figure for each L0 value
        for L0, L0_results in results_by_L0.items():
            self.create_L0_plot(L0, L0_results)
        
        # Create overall summary plot
        self.create_summary_plot(all_results)

    def create_L0_plot(self, L0, L0_results):
        """Create plot for a specific L0 value"""
        if not L0_results:
            return
            
        # Create figure with subplots for each combination
        n_cols = 3  # 3 columns (for 3 diameters)
        n_rows = 3  # 3 rows (for 3 pixel_pupils)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
        fig.suptitle(f'RMS Modal Coefficients (L0 = {L0:.0f}m): AtmoEvolution vs AtmoInfiniteEvolution', fontsize=14)
        
        # Organize results by pixel_pupil and diameter
        result_grid = {}
        for result in L0_results:
            pp = result['combo']['pixel_pupil']
            diam = result['combo']['diameter']
            result_grid[(pp, diam)] = result
        
        # Plot each combination
        for row, pixel_pupil in enumerate(self.pixel_pupils):
            for col, diameter in enumerate(self.pupil_diameters):
                ax = axes[row, col]
                
                if (pixel_pupil, diameter) in result_grid:
                    result = result_grid[(pixel_pupil, diameter)]
                    combo = result['combo']
                    
                    # Plot RMS
                    x = np.arange(len(result['rms_modes1'])) + 2  # Zernike modes start from 2
                    ax.loglog(x, result['rms_modes1'], 'b-o', markersize=3, 
                             label='AtmoEvolution', alpha=0.7)
                    ax.loglog(x, result['rms_modes2'], 'r-s', markersize=3, 
                             label='AtmoInfiniteEvolution', alpha=0.7)
                    
                    # Calculate sampling ratio for subtitle
                    D_phys = combo['diameter']
                    sampling_ratio = D_phys / combo['L0']
                    
                    # Formatting
                    ax.set_title(f'pp={pixel_pupil}, D={diameter:.0f}m\n'
                               f'pitch={combo["pixel_pitch"]:.4f}m\n'
                               f'D/L0={sampling_ratio:.2f}', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    if row == n_rows - 1:  # Bottom row
                        ax.set_xlabel('Zernike Mode')
                    if col == 0:  # Left column
                        ax.set_ylabel('RMS [rad]')
                    
                    # Add legend only to first subplot
                    if row == 0 and col == 0:
                        ax.legend(fontsize=8)
                        
                else:
                    # Empty subplot for failed combinations
                    ax.text(0.5, 0.5, f'pp={pixel_pupil}\nD={diameter:.0f}m\nFAILED', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()

    def create_summary_plot(self, all_results):
        """Create summary comparison plot showing ratios between methods"""
        if len(all_results) < 2:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        
        # Plot 1: RMS comparison for first few modes
        n_modes_show = min(20, min(len(r['rms_modes1']) for r in all_results))
        x = np.arange(n_modes_show) + 2
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_results)))
        
        for i, result in enumerate(all_results):
            combo = result['combo']
            label = f"{combo['name']}"
            
            ax1.loglog(x, result['rms_modes1'][:n_modes_show], 
                      color=colors[i], linestyle='-', alpha=0.6, linewidth=1)
            ax1.loglog(x, result['rms_modes2'][:n_modes_show], 
                      color=colors[i], linestyle='--', alpha=0.6, linewidth=1)
        
        ax1.set_xlabel('Zernike Mode')
        ax1.set_ylabel('RMS [rad]')
        ax1.set_title('RMS Comparison (first 20 modes)\nSolid: AtmoEvolution, Dashed: AtmoInfiniteEvolution')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Ratio between methods
        for i, result in enumerate(all_results):
            combo = result['combo']
            ratio = result['rms_modes2'] / result['rms_modes1']
            x_ratio = np.arange(len(ratio)) + 2
            
            ax2.semilogx(x_ratio, ratio, color=colors[i], marker='o', markersize=2,
                        label=combo['name'], alpha=0.6, linewidth=1)
        
        ax2.axhline(y=1.0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Zernike Mode')
        ax2.set_ylabel('Ratio (AtmoInfiniteEvolution / AtmoEvolution)')
        ax2.set_title('RMS Ratio Between Methods')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Effect of L0 on ratio (for tip/tilt mode - mode 2)
        L0_values = sorted(set(r['combo']['L0'] for r in all_results))
        pixel_pupil_values = sorted(set(r['combo']['pixel_pupil'] for r in all_results))
        diameter_values = sorted(set(r['combo']['diameter'] for r in all_results))
        
        for pp in pixel_pupil_values:
            for diam in diameter_values:
                ratios_vs_L0 = []
                L0_list = []
                for L0 in L0_values:
                    matching_results = [r for r in all_results 
                                      if r['combo']['pixel_pupil'] == pp 
                                      and r['combo']['diameter'] == diam 
                                      and r['combo']['L0'] == L0]
                    if matching_results:
                        result = matching_results[0]
                        ratio_mode2 = result['rms_modes2'][0] / result['rms_modes1'][0]  # Mode 2 (tip)
                        ratios_vs_L0.append(ratio_mode2)
                        L0_list.append(L0)
                
                if len(ratios_vs_L0) > 1:
                    ax3.plot(L0_list, ratios_vs_L0, 'o-', 
                            label=f'pp={pp}, D={diam:.0f}m', alpha=0.7, linewidth=2)
        
        ax3.axhline(y=1.0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('L0 [m]')
        ax3.set_ylabel('Ratio for Mode 2 (Tip)')
        ax3.set_title('Effect of L0 on Ratio (Mode 2)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        
        # Plot 4: Effect of diameter/L0 ratio on overall performance
        sampling_ratios = []
        mean_ratios = []
        
        for result in all_results:
            combo = result['combo']
            D_phys = combo['diameter']
            sampling_ratio = D_phys / combo['L0']
            
            # Calculate mean ratio for first 10 modes
            ratio_modes = result['rms_modes2'][:10] / result['rms_modes1'][:10]
            mean_ratio = np.mean(ratio_modes)
            
            sampling_ratios.append(sampling_ratio)
            mean_ratios.append(mean_ratio)
        
        ax4.scatter(sampling_ratios, mean_ratios, alpha=0.6, s=50)
        ax4.axhline(y=1.0, color='black', linestyle='-', alpha=0.5)
        ax4.set_xlabel('D/L0 Sampling Ratio')
        ax4.set_ylabel('Mean Ratio (first 10 modes)')
        ax4.set_title('Performance vs Sampling Ratio')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    unittest.main()