import unittest

import specula
specula.init(0)  # Default target device

from specula import np
from specula.data_objects.recmat import Recmat
from specula.processing_objects.mirror_commands_combinator import MirrorCommandsCombinator
from test.specula_testlib import cpu_and_gpu


class TestMirrorCommandsCombinator(unittest.TestCase):

    @cpu_and_gpu
    def test_k_vector_dtype_follows_precision(self, target_device_idx, xp):
        """k_vector must be cast to the object's own dtype, not keep the input's dtype."""
        recmat = Recmat(xp.zeros((4, 3)), target_device_idx=target_device_idx)
        k_vector = np.array([0.5, 0.3, 0.1], dtype=np.float64)

        comb32 = MirrorCommandsCombinator(
            k_vector=k_vector, recmat=recmat,
            dims_LO=[1, 0, 1], dims_P=1, dims_F=1,
            out_dims=[4, 2, 2],
            precision=1, target_device_idx=target_device_idx
        )
        self.assertEqual(comb32.k_vector.dtype, xp.float32)

        comb64 = MirrorCommandsCombinator(
            k_vector=k_vector, recmat=recmat,
            dims_LO=[1, 0, 1], dims_P=1, dims_F=1,
            out_dims=[4, 2, 2],
            precision=0, target_device_idx=target_device_idx
        )
        self.assertEqual(comb64.k_vector.dtype, xp.float64)
