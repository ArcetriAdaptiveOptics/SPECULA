import numpy as np
import unittest
import pytest

import specula
specula.init(0)

from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc


class TestComputeZonalIfunc(unittest.TestCase):

  def test_invalid_geom_raises(self):
    with pytest.raises(ValueError):
        compute_zonal_ifunc(dim=32, n_act=4, geom='not_a_geom', xp=np)
      
  def test_double_input_raises(self):
    with pytest.raises(ValueError):
        compute_zonal_ifunc(dim=32, n_act=4, circ_geom=True, geom='circular', xp=np)

  def test_circular_geom(self):
      ifs_cube,_ = compute_zonal_ifunc(dim=32, n_act=3, geom='circular')
      n_act_tot = np.shape(ifs_cube)[0]
      if n_act_tot != 19:
          raise ValueError()

  def test_square_geom(self):
      n_act = 4
      ifs_cube,_ = compute_zonal_ifunc(dim=32, n_act=n_act, geom='square')
      n_act_tot = np.shape(ifs_cube)[0]
      if n_act_tot != n_act**2:
          raise ValueError()

  def test_square_geom(self):
      n_act = 4
      ifs_cube,_ = compute_zonal_ifunc(dim=32, n_act=n_act, geom='alpao')
      n_act_tot = np.shape(ifs_cube)[0]
      if n_act_tot >= n_act**2:
          raise ValueError()
    
