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

    @cpu_and_gpu
    def test_combination(self, target_device_idx, xp):
        """Outputs are the documented combination of the inputs, including recmat @ x_P"""
        from specula import cpuArray
        from specula.base_value import BaseValue
        rng = np.random.default_rng(3)
        A = rng.standard_normal((6, 2))
        recmat = Recmat(xp.asarray(A), target_device_idx=target_device_idx, precision=0)
        comb = MirrorCommandsCombinator(k_vector=[0.5], recmat=recmat,
                                        dims_LO=[1, 0, 1], dims_P=2, dims_F=1,
                                        out_dims=[4, 2, 2], precision=0,
                                        target_device_idx=target_device_idx)
        self.assertTrue(comb.can_accelerate())
        x = {name: rng.standard_normal(n) for name, n in {'HO': 5, 'LO': 2, 'F': 1, 'P': 2}.items()}
        for name, v in x.items():
            value = BaseValue(value=xp.asarray(v), target_device_idx=target_device_idx, precision=0)
            value.generation_time = 0
            comb.inputs[f'in_commands{name}'].set(value)
        comb.setup()
        comb.check_ready(0)
        comb.trigger()
        comb.post_trigger()
        HO, LO, F, P = x['HO'], x['LO'], x['F'], x['P']
        y1 = np.array([LO[0], F[0] + 0.5 * LO[1], HO[0], HO[1]]) + A[:4] @ P
        np.testing.assert_allclose(cpuArray(comb.outputs['out_result_commands1'].value), y1)
        np.testing.assert_allclose(cpuArray(comb.outputs['out_result_commands2'].value), HO[2:4])
        np.testing.assert_allclose(cpuArray(comb.outputs['out_result_commands3'].value), [LO[1], HO[4]])


@unittest.skipIf(specula.cp is None, 'Cupy not found')
class TestMirrorCommandsCombinatorAccelerated(unittest.TestCase):
    '''The product uses graph_matmul.matmul(): the combinator is captured in a CUDA graph'''

    def _run(self, accelerate, matrix_dtype, input_dtype):
        from specula import cpuArray
        from specula.base_value import BaseValue
        from iteras.accelerator import Accelerator
        from iteras.data_objects.accelerator import Accelerator as AcceleratorOptions
        rng = np.random.default_rng(2)
        recmat = Recmat(rng.standard_normal((6, 2)).astype(matrix_dtype), target_device_idx=0,
                        precision=int(matrix_dtype == np.float32))
        comb = MirrorCommandsCombinator(k_vector=[0.5], recmat=recmat,
                                        dims_LO=[1, 0, 1], dims_P=2, dims_F=1,
                                        out_dims=[4, 2, 2], target_device_idx=0, precision=1)
        sizes = {'HO': 5, 'LO': 2, 'F': 1, 'P': 2}
        values = {name: BaseValue(value=np.zeros(n, dtype=input_dtype), target_device_idx=0,
                                   precision=int(input_dtype == np.float32))
                  for name, n in sizes.items()}
        for name, value in values.items():
            comb.inputs[f'in_commands{name}'].set(value)
        if accelerate:
            Accelerator(AcceleratorOptions()).apply([comb])
        comb.setup()
        outputs = []
        for step in range(5):
            t = comb.seconds_to_t(0.001 * step)
            for name, value in values.items():
                value.value[:] = specula.cp.asarray(rng.standard_normal(sizes[name]))
                value.generation_time = t
            comb.check_ready(t)
            comb.trigger()
            comb.post_trigger()
            outputs.append(np.concatenate([cpuArray(comb.outputs[f'out_result_commands{i}'].value)
                                           for i in (1, 2, 3)]))
        return np.array(outputs), comb

    def test_accelerated(self):
        # Also with a matrix or inputs in double precision, used in the single precision of the object
        for matrix_dtype, input_dtype in ((np.float32, np.float32),
                                          (np.float64, np.float32),
                                          (np.float32, np.float64)):
            with self.subTest(matrix_dtype=matrix_dtype, input_dtype=input_dtype):
                ref, _ = self._run(False, matrix_dtype, input_dtype)
                with self.assertNoLogs('iteras.accelerator', level='WARNING'):
                    out, comb = self._run(True, matrix_dtype, input_dtype)
                np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
                self.assertEqual(comb._accel.status, 'accelerated')
