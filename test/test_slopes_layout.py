import specula
specula.init(0)  # Default target device

import os
import pickle
import shutil
import tempfile
import unittest
import warnings

import numpy as np

from specula import cpuArray, cp
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.pupdata import PupData
from specula.data_objects.slopes import Slopes
from specula.lib.slopes_layout import SlopesLayout
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.data_source import DataSource
from specula.processing_objects.data_store import DataStore
from specula.processing_objects.zwfs_slopec import ZwfsSlopec

from test.specula_testlib import cpu_and_gpu

try:
    import h5py  # noqa: F401
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def _old_get2d(vector, mask_shape, idx, interleave):
    '''The algorithm of Slopes.get2d() before SlopesLayout, as a reference'''
    vector = np.asarray(vector)
    if vector.size == len(idx):
        f = np.zeros(mask_shape)
        np.put(f, idx, vector)
        return f
    n = vector.size
    if interleave:
        ix, iy = np.arange(n // 2) * 2, np.arange(n // 2) * 2 + 1
    else:
        ix, iy = np.arange(n // 2), np.arange(n // 2, n)
    fx = np.zeros(mask_shape)
    fy = np.zeros(mask_shape)
    np.put(fx, idx, vector[ix])
    np.put(fy, idx, vector[iy])
    return np.array([fx, fy])


class TestSlopesLayout(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # SlopesLayout ----------------------------------------------------------

    @cpu_and_gpu
    def test_to_2d_matches_previous_get2d(self, target_device_idx, xp):
        idx = np.array([1, 2, 5, 7, 8])
        for interleave in (False, True):
            layout = SlopesLayout((3, 3), idx, SlopesLayout.XY)
            vector = np.arange(1, 11, dtype=np.float32)
            got = layout.to_2d(xp.asarray(vector), xp, interleave=interleave)
            self.assertEqual(got.shape, (2, 3, 3))
            np.testing.assert_array_equal(cpuArray(got), _old_get2d(vector, (3, 3), idx, interleave))

        layout = SlopesLayout((3, 3), idx, SlopesLayout.SCALAR)
        vector = np.arange(1, 6, dtype=np.float32)
        got = layout.to_2d(xp.asarray(vector), xp)
        self.assertEqual(got.shape, (3, 3))
        np.testing.assert_array_equal(cpuArray(got), _old_get2d(vector, (3, 3), idx, False))

    @cpu_and_gpu
    def test_from_2d_is_inverse_of_to_2d(self, target_device_idx, xp):
        idx = np.array([0, 4, 5, 11])
        for kind, n in ((SlopesLayout.XY, 8), (SlopesLayout.SCALAR, 4)):
            for interleave in (False, True):
                layout = SlopesLayout((3, 4), idx, kind)
                vector = xp.arange(n, dtype=xp.float32) + 1
                frames = layout.to_2d(vector, xp, interleave=interleave)
                back = layout.from_2d(frames, xp, interleave=interleave)
                np.testing.assert_array_equal(cpuArray(back), cpuArray(vector))
                flat = frames.reshape(-1)[layout.value_idx(xp, interleave)]
                np.testing.assert_array_equal(cpuArray(flat), cpuArray(vector))

    def test_invalid_layouts(self):
        with self.assertRaises(ValueError):
            SlopesLayout((2, 2), [0, 4], SlopesLayout.XY)     # index out of range
        with self.assertRaises(ValueError):
            SlopesLayout((2, 2), [-1, 0], SlopesLayout.XY)    # negative index
        with self.assertRaises(ValueError):
            SlopesLayout((2, 2), [0, 1], 'xyz')               # unknown kind
        with self.assertRaises(ValueError):
            SlopesLayout((4,), [0, 1], SlopesLayout.XY)       # not 2d
        layout = SlopesLayout((2, 2), [0, 3], SlopesLayout.XY)
        with self.assertRaises(ValueError):
            layout.to_2d(np.zeros(3))                         # wrong length
        with self.assertRaises(ValueError):
            layout.from_2d(np.zeros((2, 2)))                  # wrong frame shape

    def test_layout_is_read_only(self):
        layout = SlopesLayout((2, 2), [0, 3], SlopesLayout.XY)
        with self.assertRaises(ValueError):
            layout.idx[0] = 1

    def test_full_frame(self):
        layout = SlopesLayout.full_frame((2, 3))
        self.assertEqual(layout.kind, SlopesLayout.SCALAR)
        self.assertEqual(layout.n_values, 6)
        vector = np.arange(6.0)
        np.testing.assert_array_equal(layout.to_2d(vector), vector.reshape(2, 3))

    def test_arrays_roundtrip_and_equality(self):
        layout = SlopesLayout((3, 5), [1, 7, 9], SlopesLayout.XY)
        arrays = layout.to_arrays()
        self.assertEqual(SlopesLayout.from_arrays(arrays), layout)
        # Names may come back in upper or lower case, with other arrays
        lower = {name.lower(): value for name, value in arrays.items()}
        lower['data'] = np.zeros(3)
        self.assertEqual(SlopesLayout.from_arrays(lower), layout)
        self.assertIsNone(SlopesLayout.from_arrays({'data': np.zeros(3)}))
        self.assertNotEqual(layout, SlopesLayout((3, 5), [1, 7, 9], SlopesLayout.SCALAR))
        self.assertNotEqual(layout, SlopesLayout((3, 5), [1, 7, 8], SlopesLayout.XY))

    @cpu_and_gpu
    def test_pickle_drops_device_cache(self, target_device_idx, xp):
        layout = SlopesLayout((2, 2), [0, 3], SlopesLayout.XY)
        layout.value_idx(xp, interleave=True)
        restored = pickle.loads(pickle.dumps(layout))
        self.assertEqual(restored, layout)
        self.assertEqual(restored._device_idx, {})
        self.assertFalse(restored.idx.flags.writeable)

    # Geometry objects ------------------------------------------------------

    @cpu_and_gpu
    def test_subapdata_layout(self, target_device_idx, xp):
        subapdata = SubapData(idxs=np.zeros((3, 4)), display_map=np.array([1, 2, 6]),
                              nx=3, ny=3, target_device_idx=target_device_idx)
        layout = subapdata.slopes_layout()
        self.assertEqual(layout.kind, SlopesLayout.XY)
        self.assertEqual(layout.shape, cpuArray(subapdata.single_mask()).shape)
        np.testing.assert_array_equal(cpuArray(layout.mask()), cpuArray(subapdata.single_mask()))
        np.testing.assert_array_equal(layout.idx, cpuArray(subapdata.display_map))

    @cpu_and_gpu
    def test_pupdata_layouts(self, target_device_idx, xp):
        # 4 pupils of 2 pixels each, one per quadrant of a 4x4 frame, pupil 0 top right
        ind_pup = np.array([[2, 0, 8, 10],
                            [3, 1, 9, 11]])
        pupdata = PupData(ind_pup=ind_pup, framesize=[4, 4], target_device_idx=target_device_idx)

        layout = pupdata.slopes_layout(from_intensity=False)
        self.assertEqual(layout.kind, SlopesLayout.XY)
        self.assertEqual(layout.shape, cpuArray(pupdata.single_mask()).shape)
        np.testing.assert_array_equal(layout.idx, cpuArray(pupdata.local_display_map()))

        layout = pupdata.slopes_layout(from_intensity=True)
        self.assertEqual(layout.kind, SlopesLayout.SCALAR)
        self.assertEqual(layout.shape, (4, 4))
        np.testing.assert_array_equal(cpuArray(layout.mask()), cpuArray(pupdata.complete_mask()))

        # Default: the setting of the PupData
        self.assertEqual(pupdata.slopes_layout().kind, SlopesLayout.XY)
        pupdata.set_slopes_from_intensity(True)
        self.assertEqual(pupdata.slopes_layout().kind, SlopesLayout.SCALAR)

    @cpu_and_gpu
    def test_zwfs_slopes_map(self, target_device_idx, xp):
        '''ZWFS slopes, one value per pupil pixel, are displayed on the whole frame'''
        pixels = Pixels(5, 5, target_device_idx=target_device_idx)
        pixels.pixels = xp.arange(25, dtype=xp.uint16).reshape((5, 5))
        pixels.generation_time = 1
        slopec = ZwfsSlopec(pup_diam=3, ccd_size=5, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(1)
        slopec.trigger()
        slopec.post_trigger()

        frame = cpuArray(slopec.slopes.get2d())
        self.assertEqual(frame.shape, (5, 5))
        np.testing.assert_array_almost_equal(frame[1:4, 1:4].ravel(), cpuArray(slopec.slopes.slopes))
        np.testing.assert_array_almost_equal(cpuArray(slopec.slopes_map.value), frame)

    # Slopes ----------------------------------------------------------------

    @cpu_and_gpu
    def test_slopes_get2d_with_layout(self, target_device_idx, xp):
        layout = SlopesLayout((2, 2), [0, 3], SlopesLayout.XY)
        slopes = Slopes(slopes=xp.arange(4.0), layout=layout, target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(slopes.get2d()),
                                      _old_get2d(np.arange(4.0), (2, 2), [0, 3], False))
        np.testing.assert_array_equal(cpuArray(slopes.single_mask), [[1, 0], [0, 1]])
        np.testing.assert_array_equal(cpuArray(slopes.display_map), [0, 3])
        self.assertEqual(slopes.array_for_display().shape, (2, 4))

        slopes = Slopes(length=4, target_device_idx=target_device_idx)
        with self.assertRaises(ValueError):
            slopes.get2d()
        self.assertIsNone(slopes.single_mask)
        self.assertIsNone(slopes.display_map)

    @cpu_and_gpu
    def test_scalar_array_for_display(self, target_device_idx, xp):
        slopes = Slopes(slopes=xp.arange(6.0), layout=SlopesLayout.full_frame((2, 3)),
                        target_device_idx=target_device_idx)
        self.assertEqual(slopes.array_for_display().shape, (2, 3))

    @cpu_and_gpu
    def test_legacy_mask_and_map(self, target_device_idx, xp):
        slopes = Slopes(slopes=xp.arange(4.0), target_device_idx=target_device_idx)
        with self.assertWarns(DeprecationWarning):
            slopes.single_mask = xp.zeros((2, 2))
        self.assertIsNone(slopes.layout)
        with self.assertWarns(DeprecationWarning):
            slopes.display_map = xp.array([0, 3])
        self.assertEqual(slopes.layout, SlopesLayout((2, 2), [0, 3], SlopesLayout.XY))

        # Slopes from intensity: as many values as indices
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            slopes.display_map = xp.array([0, 1, 2, 3])
        self.assertEqual(slopes.layout.kind, SlopesLayout.SCALAR)

        # 2d (row, column) indices
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            slopes.display_map = xp.array([[0, 1], [0, 1]])
        np.testing.assert_array_equal(slopes.layout.idx, [0, 3])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            slopes.display_map = None
        self.assertIsNone(slopes.layout)

    @cpu_and_gpu
    def test_layout_is_shared_not_copied(self, target_device_idx, xp):
        layout = SlopesLayout((2, 2), [0, 3], SlopesLayout.XY)
        slopes = Slopes(slopes=xp.arange(4.0), layout=layout, target_device_idx=target_device_idx)
        self.assertNotIn('layout', [k for k, v in vars(slopes).items()
                                    if isinstance(v, (np.ndarray,) + ((cp.ndarray,) if cp else ()))])
        other_device = 0 if target_device_idx == -1 else -1
        if other_device == 0 and cp is None:
            return
        copied = slopes.copyTo(other_device)
        self.assertIs(copied.layout, layout)
        dest = Slopes(length=4, target_device_idx=other_device)
        slopes.transferDataTo(dest)
        self.assertIs(dest.layout, layout)

    @cpu_and_gpu
    def test_save_restore_keeps_layout(self, target_device_idx, xp):
        layout = SlopesLayout((3, 3), [1, 4, 7], SlopesLayout.XY)
        slopes = Slopes(slopes=xp.arange(6.0), layout=layout, interleave=True,
                        target_device_idx=target_device_idx)
        for ext in ('.fits', '.npz', '.npy') + (('.h5',) if HAS_H5PY else ()):
            filename = os.path.join(self.tmp_dir, 'slopes' + ext)
            slopes.save(filename)
            restored = Slopes.restore(filename, target_device_idx=target_device_idx)
            self.assertEqual(restored.layout, layout, ext)
            np.testing.assert_array_equal(cpuArray(restored.get2d()), cpuArray(slopes.get2d()))

        # Without a layout, as in files written by older versions
        filename = os.path.join(self.tmp_dir, 'no_layout.fits')
        Slopes(slopes=xp.arange(6.0), target_device_idx=target_device_idx).save(filename)
        self.assertIsNone(Slopes.restore(filename).layout)

    # DataStore / DataSource ------------------------------------------------

    def _store_and_reload(self, slopes, data_format):
        store_dir = os.path.join(self.tmp_dir, data_format)
        os.mkdir(store_dir)
        store = DataStore(store_dir=store_dir, data_format=data_format, create_tn=False)
        store.inputs['sl'] = InputValue(type=Slopes)
        store.inputs['sl'].set(slopes)
        store.setup()
        for t in (0, 1):
            slopes.slopes[:] = slopes.xp.arange(slopes.size) + t
            slopes.generation_time = t
            store.check_ready(t)
            store.trigger()
            store.post_trigger()
        if data_format == 'pickle':
            store.save_pickle()
        else:
            store.save_data()
        return DataSource(outputs=['sl'], store_dir=store_dir, data_format=data_format)

    @cpu_and_gpu
    def test_data_store_source_keep_layout(self, target_device_idx, xp):
        layout = SlopesLayout((3, 3), [1, 4, 7], SlopesLayout.XY)
        formats = ['fits', 'npz', 'npy', 'pickle'] + (['hdf5'] if HAS_H5PY else [])
        for data_format in formats:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            os.mkdir(self.tmp_dir)
            slopes = Slopes(length=6, layout=layout, target_device_idx=target_device_idx)
            source = self._store_and_reload(slopes, data_format)
            out = source.outputs['sl']
            self.assertEqual(out.layout, layout, data_format)
            np.testing.assert_array_equal(cpuArray(source.storage['sl'][1]), np.arange(6) + 1)

    def test_data_store_without_layout(self):
        slopes = Slopes(length=6, target_device_idx=-1)
        source = self._store_and_reload(slopes, 'fits')
        self.assertIsNone(source.outputs['sl'].layout)
        self.assertEqual(source.metadata['sl'], {})

    def test_reserved_metadata_names(self):
        class BadSlopes(Slopes):
            def get_metadata_arrays(self):
                return {'times': np.zeros(2)}
        with self.assertRaises(ValueError):
            DataStore._metadata_arrays(BadSlopes(length=2, target_device_idx=-1))


if __name__ == '__main__':
    unittest.main()
