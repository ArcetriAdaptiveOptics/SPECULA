import glob
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import yaml

from test.specula_testlib import cpu_and_gpu

try:
    import orthogram  # Check if orthogram is installed
    ORTHOGRAM_AVAILABLE = True
except ImportError:
    ORTHOGRAM_AVAILABLE = False

import specula
specula.init(0)  # Default target device

# Import your module — adjust the path if needed
from specula.simul import Simul
from specula import main_simul
from specula.simul_diagram import SimulDiagram


@unittest.skipUnless(ORTHOGRAM_AVAILABLE, "Skipping diagram tests (orthogram not installed)")
class TestDiagrams(unittest.TestCase):
    def setUp(self):
        """Set up a temp PNG file and dummy parameters for tests."""
        self.tmp_png_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.tmp_png_path = Path(self.tmp_png_file.name)
        self.tmp_png_file.close()
        self.yml_path = Path(self.tmp_png_path).with_suffix(".yml")
        self.tmp_png_path2 = Path(self.tmp_png_file.name + '2')

        # This is not really executed, just the objects are built in one of the tests
        self.dummy_params = {
            "main": { "class": "SimulParams", "root_dir": "/tmp", "total_time": 1, "time_step": 1 },
            "A": { "class": "WaveGenerator", "target_device_idx": -1, "constant": 1 },
            "B": { "class": "WaveGenerator", "target_device_idx": -1, "constant": 2 },
            "C": { "class": "WaveGenerator", "target_device_idx": -1, "constant": 3 },
        }

        self.calibdir = os.path.join(os.path.dirname(__file__), 'calib')
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.outputdir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.datadir, exist_ok=True)
        os.makedirs(self.outputdir, exist_ok=True)
        self.phasescreen_path = os.path.join(self.calibdir, 'phasescreens',
                                   'ps_seed1_dim2048_pixpit0.301_L025.0000_single.fits')
        self.cwd = os.getcwd()

    def tearDown(self):
        """Clean up the temp files."""
        try:
            os.remove(self.tmp_png_path)
        except FileNotFoundError:
            pass

        try:
            os.remove(self.yml_path)
        except FileNotFoundError:
            pass

        # Clean up output directories created by the simulation
        data_dirs = glob.glob(os.path.join(self.outputdir, '2*'))
        for data_dir in data_dirs:
            if os.path.isdir(data_dir):
                shutil.rmtree(data_dir)
        ps_dir = os.path.dirname(self.phasescreen_path)
        ps_base = os.path.basename(self.phasescreen_path).replace('_single.fits', '_*.fits')
        for fpath in glob.glob(os.path.join(ps_dir, ps_base)):
            os.remove(fpath)
        os.chdir(self.cwd)

    def _make_diagram(self, colors=False):
        """Helper to create a Simul instance configured for diagram tests."""
        diagram = SimulDiagram(param_file="dummy.yml",
                               title="Test Diagram",
                               filename=str(self.tmp_png_path),
                               colors_on=colors)
        diagram.build(trigger_order = ["A", "B", "C"],
                    trigger_order_idx = [0, 1, 2],
                    all_objs_ranks = {"A": 0, "B": 1, "C": 0},  
                    all_target_device_idxs = {"A": 0, "B": 0, "C": 0},
                    is_dataobj = {"A": True, "B": False, "C": True},
        )
        return diagram

    @patch("orthogram.write_png")
    def test_build_diagram_basic(self, mock_write_png):
        """Test that buildDiagram() creates a diagram and calls write_png."""
        diagram = self._make_diagram(colors=False)
        mock_write_png.assert_called_once()
        args, kwargs = mock_write_png.call_args
        self.assertEqual(str(self.tmp_png_path), str(args[1]))

    @patch("orthogram.write_png")
    def test_build_diagram_with_colors(self, mock_write_png):
        """Test diagram creation with colors enabled."""
        diagram = self._make_diagram(colors=True)
        mock_write_png.assert_called_once()
        args, kwargs = mock_write_png.call_args
        self.assertIn(".png", str(args[1]))

    def test_main_simul_with_diagram(self):
        """Test main_simul() triggers diagram generation when enabled."""
        with open(self.yml_path, "w") as f:
            yaml.dump(self.dummy_params, f)

        with patch("orthogram.write_png") as mock_write_png:
            main_simul(
                yml_files=[str(self.yml_path)],
                nsimul=1,
                cpu=True,
                diagram=True,
            )

        mock_write_png.assert_called()

    def test_main_simul_with_diagram_and_filename(self):
        """Test main_simul() triggers diagram generation when enabled."""
        yml_path = Path(self.tmp_png_path).with_suffix(".yml")
        with open(self.yml_path, "w") as f:
            yaml.dump(self.dummy_params, f)

        with patch("orthogram.write_png") as mock_write_png:
            main_simul(
                yml_files=[str(self.yml_path)],
                nsimul=1,
                cpu=True,
                diagram=True,
                diagram_filename=str(self.tmp_png_path2),
                diagram_title="MainSimul Diagram",
                diagram_colors_on=True,
            )

        mock_write_png.assert_called()
        assert mock_write_png.call_args.args[1] == str(self.tmp_png_path2)


    def test_real_diagram_creation(self):
        """Integration test: diagram creation in a full simulation run"""
        os.chdir(os.path.dirname(__file__))

        yml_files = ['params_elt_pfs_test.yml']
        simul = Simul(*yml_files, diagram_filename=self.tmp_png_path)
        simul.run()
        assert os.path.exists(self.tmp_png_path)

