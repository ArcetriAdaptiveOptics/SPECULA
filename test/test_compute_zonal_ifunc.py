import numpy as np
import unittest
import pytest

import specula
specula.init(0)

from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
from test.specula_testlib import cpu_and_gpu


class TestComputeZonalIfunc(unittest.TestCase):

  @cpu_and_gpu
  def test_invalid_geom_raises(self, xp):
    with pytest.raises(ValueError):
        compute_zonal_ifunc(dim=32, n_act=4, geom='not_a_geom', xp=xp, dtype=xp.float32)
      
  @cpu_and_gpu
  def test_double_input_raises(self, xp):
    with pytest.raises(ValueError):
        compute_zonal_ifunc(dim=32, n_act=4, circ_geom=True, geom='circular', xp=xp, dtype=xp.float32)
      
  @cpu_and_gpu
  def test_circular_geom(self, xp):
      ifs_cube,_ = compute_zonal_ifunc(dim=32, n_act=3, geom='circular', xp=xp, dtype=xp.float32)
      n_act_tot = np.shape(ifs_cube)[0]
      if n_act_tot != 19:
          raise ValueError()

  @cpu_and_gpu
  def test_square_geom(self, xp):
      n_act = 4
      ifs_cube,_ = compute_zonal_ifunc(dim=32, n_act=n_act, geom='square', xp=xp, dtype=xp.float32)
      n_act_tot = np.shape(ifs_cube)[0]
      if n_act_tot != n_act**2:
          raise ValueError()

  @cpu_and_gpu
  def test_square_geom(self, xp):
      n_act = 4
      ifs_cube,_ = compute_zonal_ifunc(dim=32, n_act=n_act, geom='alpao', xp=xp, dtype=xp.float32)
      n_act_tot = np.shape(ifs_cube)[0]
      if n_act_tot >= n_act**2:
          raise ValueError()
    
