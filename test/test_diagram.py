import sys
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import yaml

try:
    import orthogram  # Check if orthogram is installed
    ORTHOGRAM_AVAILABLE = True
except ImportError:
    ORTHOGRAM_AVAILABLE = False

# Import your module — adjust the path if needed
from specula.simul import Simul
from specula import main_simul


@unittest.skipUnless(ORTHOGRAM_AVAILABLE, "Skipping diagram tests (orthogram not installed)")
class TestDiagrams(unittest.TestCase):
    def setUp(self):
        """Set up a temp PNG file and dummy parameters for tests."""
        self.tmp_png_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.tmp_png_path = Path(self.tmp_png_file.name)
        self.tmp_png_file.close()

        # This is not really executed, just the objects are built in one of the tests
        self.dummy_params = {
            "main": { "class": "SimulParams", "root_dir": "/tmp", "total_time": 1, "time_step": 1 },
            "A": { "class": "WaveGenerator", "target_device_idx": -1, "constant": 1 },
            "B": { "class": "WaveGenerator", "target_device_idx": -1, "constant": 2 },
            "C": { "class": "WaveGenerator", "target_device_idx": -1, "constant": 3 },
        }

    def tearDown(self):
        """Clean up the temp file."""
        try:
            os.remove(self.tmp_png_path)
        except FileNotFoundError:
            pass

    def _make_simul(self, colors=False):
        """Helper to create a Simul instance configured for diagram tests."""
        simul = Simul("dummy.yml")
        simul.trigger_order = ["A", "B", "C"]
        simul.trigger_order_idx = [0, 1, 2]
        simul.all_objs_ranks = {"A": 0, "B": 1, "C": 0}
        simul.max_rank = 1
        simul.max_target_device_idx = 1
        simul.is_dataobj = {"A": True, "B": False, "C": True}
        simul.connections = []
        simul.references = []
        simul.diagram_filename = str(self.tmp_png_path)
        simul.diagram_title = "Test Diagram"
        simul.diagram_colors_on = colors
        return simul

    @patch("orthogram.write_png")
    def test_build_diagram_basic(self, mock_write_png):
        """Test that buildDiagram() creates a diagram and calls write_png."""
        simul = self._make_simul(colors=False)
        simul.buildDiagram(self.dummy_params)
        mock_write_png.assert_called_once()
        args, kwargs = mock_write_png.call_args
        self.assertEqual(str(self.tmp_png_path), str(args[1]))

    @patch("orthogram.write_png")
    def test_build_diagram_with_colors(self, mock_write_png):
        """Test diagram creation with colors enabled."""
        simul = self._make_simul(colors=True)
        simul.buildDiagram(self.dummy_params)
        mock_write_png.assert_called_once()
        args, kwargs = mock_write_png.call_args
        self.assertIn(".png", str(args[1]))

    @patch("orthogram.write_png")
    def test_diagram_title_and_filename(self, mock_write_png):
        """Verify custom diagram title and filename handling."""
        simul = self._make_simul()
        simul.diagram_title = "Custom Title"
        simul.diagram_filename = str(self.tmp_png_path)
        simul.buildDiagram(self.dummy_params)
        mock_write_png.assert_called_once()
        args, kwargs = mock_write_png.call_args
        assert simul.diagram_filename in args

    def test_main_simul_with_diagram(self):
        """Test main_simul() triggers diagram generation when enabled."""
        yml_path = Path(self.tmp_png_path).with_suffix(".yml")
        with open(yml_path, "w") as f:
            yaml.dump(self.dummy_params, f)

        with patch("orthogram.write_png") as mock_write_png:
            main_simul(
                yml_files=[str(yml_path)],
                nsimul=1,
                cpu=True,
                diagram=True,
                diagram_filename=str(self.tmp_png_path),
                diagram_title="MainSimul Diagram",
                diagram_colors_on=True,
            )

        mock_write_png.assert_called()
        os.remove(yml_path)


