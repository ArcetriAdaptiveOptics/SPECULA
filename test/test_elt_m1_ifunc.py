
import specula
specula.init(0)  # Default target device

import os
import numpy as np
import pytest
import unittest

from specula.lib.ELT_M1_ifunc_calculator import ELTM1IFuncCalculator
from specula import cpu_float_dtype_list

class TestELTM1IFuncCalculator(unittest.TestCase):

    def setUp(self):
        self.dim = 64  # Use a small dimension for fast testing
        self.calculator = ELTM1IFuncCalculator(dim=self.dim,dtype=cpu_float_dtype_list[1])

    @pytest.mark.filterwarnings('ignore:.*divide by zero encountered*:RuntimeWarning')
    @pytest.mark.filterwarnings('ignore:.*invalid value encountered*:RuntimeWarning')
    def test_modal_base_shapes(self):
        self.calculator.M1_modal_base()
        print("size of M1 modal base: ", self.calculator.ifs_cube.shape)
        # Check that ifs_cube and mask are not None
        self.assertIsNotNone(self.calculator.ifs_cube)
        self.assertIsNotNone(self.calculator.mask)
        # Check mask shape
        self.assertEqual(self.calculator.mask.shape, (self.dim, self.dim))
        # Check ifs_cube shape: (number of pupil pixels, number of modes)
        n_modes = self.calculator.ifs_cube.shape[0]
        dim = self.calculator.mask.shape[0]
        self.assertEqual(n_modes,3*798)
        self.assertEqual(dim, self.dim)
        self.assertEqual(self.calculator.ifs_cube.shape[1], np.count_nonzero(self.calculator.mask))

    def test_save_mask(self):
        filename = "test_mask_elt_m1.fits"
        try:
            self.calculator.save_mask(filename)
            self.assertTrue(os.path.exists(filename))
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    @pytest.mark.filterwarnings('ignore:.*divide by zero encountered*:RuntimeWarning')
    @pytest.mark.filterwarnings('ignore:.*invalid value encountered*:RuntimeWarning')
    def test_save_results(self):
        filename = "test_ifunc_elt_m1.fits"
        try:
            self.calculator.save_results(filename)
            self.assertTrue(os.path.exists(filename))
        finally:
            if os.path.exists(filename):
                os.remove(filename)