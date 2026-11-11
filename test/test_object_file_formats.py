'''
Data objects can be saved and restored in npz, npy and HDF5 files, besides
FITS (see iteras.lib.storage).
'''
import specula
specula.init(0)  # Default target device

import contextlib
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np

from iteras.lib import storage
from iteras.calib_manager import CalibManager
from specula import cpuArray
from specula.data_objects.ifunc import IFunc
from specula.data_objects.recmat import Recmat

FORMATS = ['npz', 'npy', 'hdf5']

TEST_DIR = Path(__file__).resolve().parent


def _env_without_format():
    return {k: v for k, v in os.environ.items()
            if k not in ('ITERAS_DATA_FORMAT', 'SPECULA_DATA_FORMAT')}


class TestAllClasses(unittest.TestCase):

    def test_existing_save_tests_in_all_formats(self):
        '''
        Run the tests that save data objects with ITERAS_TEST_CHECK_OBJECT_FORMATS=1
        (see conftest.py): each object saved to FITS is also saved to npz, npy and
        HDF5, and the objects restored from the four files must be equal.
        '''
        files = sorted(str(p) for p in TEST_DIR.glob('test_*.py')
                       if p.name != Path(__file__).name and '.save(' in p.read_text(encoding='utf-8'))
        env = dict(os.environ, ITERAS_TEST_CHECK_OBJECT_FORMATS='1')
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *files],
                           capture_output=True, text=True, env=env, cwd=TEST_DIR.parent)
        self.assertEqual(r.returncode, 0, r.stdout[-5000:] + r.stderr[-2000:])
        match = re.search(r'Object formats checked: (.*)', r.stdout)
        self.assertIsNotNone(match, r.stdout[-2000:])
        checked = {tuple(item.strip().split('/')) for item in match.group(1).split(',')}
        classes = {name for name, _ in checked}
        for data_format in FORMATS:
            self.assertTrue({(name, data_format) for name in classes} <= checked)
        # Most data object classes are saved by their tests
        for name in ['BaseValue', 'ElectricField', 'Intmat', 'Recmat', 'Pixels', 'Slopes',
                     'SubapData', 'PupData', 'M2C', 'IFuncInv', 'IirFilterData', 'SsrFilterData',
                     'Pupilstop', 'Layer', 'ConvolutionKernel', 'SpatioTempArray']:
            self.assertIn(name, classes)

    def test_check_mode_helpers(self):
        '''
        The helpers of the check mode (conftest.py), in this process: the test
        above runs them in a subprocess, which is not measured by coverage.
        '''
        import importlib
        conftest = importlib.import_module(f'{__package__}.conftest' if __package__ else 'conftest')
        from iteras.base_data_obj import BaseDataObj
        from iteras.base_value import BaseValue
        checked = set(conftest._checked_classes)
        self.addCleanup(lambda: (conftest._checked_classes.clear(),
                                 conftest._checked_classes.update(checked)))
        conftest._checked_classes.clear()
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        recmat = Recmat(np.arange(6, dtype=np.float32).reshape(3, 2), target_device_idx=-1)

        # A FITS file is saved and checked in the other formats; other files are not checked
        save = conftest._checking_save(Recmat.save)
        save(recmat, os.path.join(tmpdir.name, 'rec.npz'))
        self.assertEqual(conftest._checked_classes, set())
        save(recmat, os.path.join(tmpdir.name, 'rec.fits'))
        self.assertEqual(conftest._checked_classes, {('Recmat', f) for f in FORMATS})

        # Classes that cannot be restored from the file alone are not checked
        class NeedsArguments(Recmat):
            @staticmethod
            def restore(filename, kernel_obj):
                raise AssertionError('not called with the file alone')
        conftest._checked_classes.clear()
        save(NeedsArguments(recmat.recmat, target_device_idx=-1), os.path.join(tmpdir.name, 'n.fits'))
        self.assertEqual(conftest._checked_classes, set())

        # Differences between the restored objects are reported
        class LosesData(Recmat):
            @staticmethod
            def restore(filename, target_device_idx=None):
                obj = Recmat.restore(filename, target_device_idx=-1)
                if not str(filename).endswith('.fits'):
                    obj.recmat[0, 0] = -1
                return obj
        with self.assertRaisesRegex(AssertionError, 'restored from npz.matrix'):
            save(LosesData(recmat.recmat, target_device_idx=-1), os.path.join(tmpdir.name, 'l.fits'))

        # Field by field comparison
        same = conftest._assert_same_restored
        same({'a': [1, (2.0, float('nan'))], 'b': np.ones(2)}, {'a': [1, (2.0, float('nan'))], 'b': np.ones(2)})
        for a, b in [(1, 2), ([1], [1, 2]), ({'a': 1}, {'b': 1}), (np.ones(2), np.ones(2, dtype=np.float32)),
                     (np.ones(2), np.zeros(2)), (recmat, BaseValue(value=np.ones(2), target_device_idx=-1))]:
            with self.subTest(a=a, b=b), self.assertRaises(AssertionError):
                same(a, b)

        # pytest_configure() wraps the save() methods of the classes defined
        # afterwards, only in check mode
        saved = BaseDataObj.__dict__.get('__init_subclass__')
        try:
            with patch.dict(os.environ, {'ITERAS_TEST_CHECK_OBJECT_FORMATS': '1'}):
                conftest.pytest_configure(None)

            class Saved(BaseDataObj):
                def save(self, filename):
                    pass
            self.assertTrue(hasattr(Saved.save, '__wrapped__'))
        finally:
            if saved is None:
                del BaseDataObj.__init_subclass__
            else:
                BaseDataObj.__init_subclass__ = saved

        # Summary of the checked classes at the end of the tests
        reporter = MagicMock()
        conftest._checked_classes.update({('Recmat', 'npz'), ('IFunc', 'npz')})
        conftest.pytest_terminal_summary(reporter)
        reporter.write_line.assert_called_once_with('Object formats checked: IFunc/npz, Recmat/npz')

    def test_ifunc_round_trip(self):
        # IFunc is not saved by its tests
        mask = np.ones((4, 4))
        mask[0, 0] = 0
        ifunc = IFunc(ifunc=np.arange(45, dtype=np.float32).reshape(3, 15), mask=mask,
                      target_device_idx=-1)
        with tempfile.TemporaryDirectory() as tmpdir:
            for data_format in FORMATS:
                with self.subTest(data_format=data_format):
                    path = os.path.join(tmpdir, 'ifunc' + storage.EXTENSIONS[data_format])
                    ifunc.save(path)
                    restored = IFunc.restore(path, target_device_idx=-1)
                    np.testing.assert_array_equal(cpuArray(restored.influence_function),
                                                  cpuArray(ifunc.influence_function))
                    np.testing.assert_array_equal(cpuArray(restored.mask_inf_func),
                                                  cpuArray(ifunc.mask_inf_func))


class TestFiles(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.recmat = Recmat(np.arange(6, dtype=np.float32).reshape(3, 2), target_device_idx=-1)

    def test_file_contents(self):
        '''The arrays and the header are stored with their names, readable without iteras'''
        expected = cpuArray(self.recmat.recmat)
        # FITS: same layout as before, with named extensions
        path = os.path.join(self.tmp.name, 'rec.fits')
        self.recmat.save(path)
        from astropy.io import fits
        with fits.open(path) as hdul:
            self.assertEqual(hdul[1].name, 'MATRIX')
            np.testing.assert_array_equal(hdul[1].data, expected)
            self.assertEqual(hdul[0].header['NORMFACT'], self.recmat.norm_factor)
        # npz
        path = os.path.join(self.tmp.name, 'rec.npz')
        self.recmat.save(path)
        with np.load(path, allow_pickle=False) as f:
            np.testing.assert_array_equal(f['matrix'], expected)
            self.assertIn('NORMFACT', str(f['__header__']))
        # npy: a directory with one file per array and the header
        path = os.path.join(self.tmp.name, 'rec.npy')
        self.recmat.save(path)
        np.testing.assert_array_equal(np.load(os.path.join(path, 'matrix.npy')), expected)
        self.assertTrue(os.path.exists(os.path.join(path, 'header.json')))
        # HDF5
        import h5py
        path = os.path.join(self.tmp.name, 'rec.h5')
        self.recmat.save(path)
        with h5py.File(path, 'r') as f:
            np.testing.assert_array_equal(f['matrix'][()], expected)
            self.assertEqual(f.attrs['NORMFACT'], self.recmat.norm_factor)

    def test_overwrite(self):
        path = os.path.join(self.tmp.name, 'rec.npz')
        self.recmat.save(path)
        with self.assertRaisesRegex(OSError, 'already exists'):
            self.recmat.save(path, overwrite=False)
        self.recmat.save(path, overwrite=True)

    def test_path_objects(self):
        path = Path(self.tmp.name) / 'rec.npz'
        self.recmat.save(path)
        restored = Recmat.restore(path, target_device_idx=-1)
        np.testing.assert_array_equal(cpuArray(restored.recmat), cpuArray(self.recmat.recmat))


class TestStorageAPI(unittest.TestCase):
    '''write_object() and read_object()'''

    def setUp(self):
        from astropy.io import fits
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.header = fits.Header()
        self.header['VERSION'] = 2
        self.header['NAME'] = ('test', 'a comment')
        self.header.add_comment('a comment card')
        self.a = np.arange(6, dtype=np.float64).reshape(2, 3)
        self.b = np.arange(4, dtype=np.int32)

    def _path(self, name, data_format):
        return os.path.join(self.tmp.name, name + storage.EXTENSIONS[data_format])

    def test_fits_layouts(self):
        from astropy.io import fits
        arrays = {'a': self.a, 'b': self.b}
        for layout, primary_shape, n_hdus in [('extensions', None, 3), ('primary', (2, 3), 2),
                                              ('placeholder', (2,), 3)]:
            with self.subTest(layout=layout):
                path = self._path(layout, 'fits')
                storage.write_object(path, self.header, arrays, layout=layout)
                with fits.open(path) as hdul:
                    self.assertEqual(len(hdul), n_hdus)
                    shape = None if hdul[0].data is None else hdul[0].data.shape
                    self.assertEqual(shape, primary_shape)
                    self.assertEqual(hdul[-1].name, 'B')
                    self.assertEqual(hdul[0].header['NAME'], 'test')
                with storage.read_object(path) as f:
                    # The array in the primary HDU has no name in FITS files
                    np.testing.assert_array_equal(f.read('a', 0 if layout == 'primary' else 1), self.a)
                    np.testing.assert_array_equal(f.read('b'), self.b)
        with self.assertRaisesRegex(ValueError, 'layout'):
            storage.write_object(self._path('x', 'fits'), self.header, arrays, layout='other')

    def test_headers_and_arrays_in_all_formats(self):
        for data_format in storage.OBJECT_FORMATS:
            with self.subTest(data_format=data_format):
                path = self._path('obj', data_format)
                storage.write_object(path, self.header, {'a': self.a, 'b': self.b, 'none': None})
                with storage.read_object(path) as f:
                    self.assertEqual(f.header['VERSION'], 2)
                    self.assertEqual(f.header['NAME'], 'test')
                    self.assertEqual(f.header.comments['NAME'], 'a comment')
                    self.assertIn('a comment card', str(f.header['COMMENT']))
                    self.assertTrue(f.has('a', 1))
                    self.assertFalse(f.has('none', 3))
                    np.testing.assert_array_equal(f.read('a', 1), self.a)
                    self.assertEqual(f.read('b', 2).dtype, np.int32)
                    self.assertTrue(f.read('a', 1).dtype.isnative)
                    with self.assertRaises(KeyError):
                        f.read('missing', 5)

    def test_old_fits_files_by_index(self):
        '''FITS files without extension names, as written by older versions'''
        from astropy.io import fits
        path = self._path('old', 'fits')
        fits.writeto(path, np.zeros(2), self.header)
        fits.append(path, self.a)
        with storage.read_object(path) as f:
            np.testing.assert_array_equal(f.read('a', 1), self.a)
            np.testing.assert_array_equal(f.read(None, 1), self.a)
            self.assertFalse(f.has('b', 2))

    def test_fits_has_does_not_read_data(self):
        path = self._path('obj', 'fits')
        storage.write_object(path, self.header, {'a': self.a, 'none': None})
        with storage.read_object(path) as f:
            # any access to the HDU data fails while has() runs
            hdu_types = {type(hdu) for hdu in f._hdul}
            with contextlib.ExitStack() as stack:
                for t in hdu_types:
                    stack.enter_context(patch.object(
                        t, 'data', new_callable=PropertyMock,
                        side_effect=AssertionError('has() read the HDU data')))
                self.assertTrue(f.has('a'))
                self.assertFalse(f.has('none'))

    def test_load_on_device(self):
        targets = [-1] + ([0] if specula.cp is not None else [])
        for data_format in storage.OBJECT_FORMATS:
            path = self._path('load', data_format)
            storage.write_object(path, self.header, {'a': self.a, 'b': self.b})
            for target in targets:
                with self.subTest(data_format=data_format, target=target), storage.read_object(path) as f:
                    a32 = f.load('a', 1, target_device_idx=target, precision=1)
                    a64 = f.load('a', 1, target_device_idx=target, precision=0)
                    b = f.load('b', 2, target_device_idx=target)
                    self.assertEqual((a32.dtype, a64.dtype, b.dtype),
                                     (np.float32, np.float64, np.int32))
                    if target >= 0:  # pragma: gpu
                        self.assertIsInstance(a32, specula.cp.ndarray)
                    np.testing.assert_array_equal(cpuArray(a64), self.a)
                    np.testing.assert_array_equal(cpuArray(b), self.b)
                    with self.assertRaisesRegex(ValueError, 'precision=1 was requested'):
                        f.load('b', 2, target_device_idx=target, precision=1)

    def test_npy_does_not_replace_other_directories(self):
        path = os.path.join(self.tmp.name, 'other.npy')
        os.makedirs(path)
        with self.assertRaisesRegex(OSError, 'not a directory written'):
            storage.write_object(path, self.header, {'a': self.a})

    def test_npy_overwrite(self):
        path = self._path('over', 'npy')
        storage.write_object(path, self.header, {'a': self.a, 'b': self.b})
        storage.write_object(path, self.header, {'b': self.b})
        with storage.read_object(path) as f:
            self.assertFalse(f.has('a'))
        self.assertFalse(os.path.exists(os.path.join(path, 'a.npy')))

    def test_no_header(self):
        for data_format in storage.OBJECT_FORMATS:
            with self.subTest(data_format=data_format):
                path = self._path('nohdr', data_format)
                storage.write_object(path, None, {'a': self.a})
                with storage.read_object(path) as f:
                    self.assertNotIn('VERSION', f.header)
                    np.testing.assert_array_equal(f.read('a', 1), self.a)

    def test_header_values_from_dictionary(self):
        '''numpy scalars are stored as numbers, other types as strings'''
        header = {'NUM': np.float32(1.5), 'LIST': [1, 2], 'NONE': None}
        for data_format in ('npz', 'npy', 'hdf5'):
            with self.subTest(data_format=data_format):
                path = self._path('dict', data_format)
                storage.write_object(path, header, {'a': self.a})
                with storage.read_object(path) as f:
                    self.assertEqual(f.header['NUM'], 1.5)
                    self.assertEqual(f.header['LIST'], '[1, 2]')

    def test_read_without_extension(self):
        '''Files without a known extension are FITS files or npy directories'''
        for data_format in ('fits', 'npy'):
            with self.subTest(data_format=data_format):
                path = os.path.join(self.tmp.name, 'noext_' + data_format)
                storage.write_object(self._path('tmp', data_format), self.header, {'a': self.a})
                os.rename(self._path('tmp', data_format), path)
                with storage.read_object(path) as f:
                    np.testing.assert_array_equal(f.read('a', 1), self.a)

    def test_load_on_default_device(self):
        for data_format in storage.OBJECT_FORMATS:
            with self.subTest(data_format=data_format):
                path = self._path('default', data_format)
                storage.write_object(path, self.header, {'a': self.a})
                with storage.read_object(path) as f:
                    np.testing.assert_array_equal(cpuArray(f.load('a', 1)), self.a)

    def test_load_missing_arrays(self):
        for data_format in storage.OBJECT_FORMATS:
            with self.subTest(data_format=data_format):
                path = self._path('missing', data_format)
                storage.write_object(path, self.header, {'a': self.a})
                with storage.read_object(path) as f:
                    with self.assertRaises(KeyError):
                        f.load('missing', 5, target_device_idx=-1)
                    if data_format == 'fits':
                        # The primary HDU has no data
                        with self.assertRaises(KeyError):
                            f.load(None, 0, target_device_idx=-1)

    def test_find_format(self):
        self.assertIsNone(storage.find_format(self.tmp.name, 'none'))
        storage.write_object(self._path('found', 'npz'), self.header, {'a': self.a})
        self.assertEqual(storage.find_format(self.tmp.name, 'found'), 'npz')

    def test_headers_without_fits_limits(self):
        '''npz, npy and HDF5 headers keep long keywords and text that is not ASCII'''
        import warnings
        header = {'WAVELENGTH_UNIT': 'µm', 'VERSION': 1}
        for data_format in ('npz', 'npy', 'hdf5'):
            with self.subTest(data_format=data_format):
                path = self._path('nonfits', data_format)
                with warnings.catch_warnings():
                    warnings.simplefilter('error')   # no HIERARCH warning
                    storage.write_object(path, header, {'a': self.a})
                    with storage.read_object(path) as f:
                        self.assertIsInstance(f.header, storage.Header)
                        self.assertEqual(f.header['WAVELENGTH_UNIT'], 'µm')
                        self.assertEqual(f.header.get('VERSION'), 1)
                    self.assertEqual(storage.read_header(path, data_format)['WAVELENGTH_UNIT'], 'µm')

    def test_header_round_trip(self):
        '''A Header read from a file is written again with its comments'''
        for data_format in ('npz', 'npy', 'hdf5'):
            with self.subTest(data_format=data_format):
                path = self._path('first', data_format)
                storage.write_object(path, self.header, {'a': self.a})
                with storage.read_object(path) as f:
                    header = f.header
                copy = self._path('copy', data_format)
                storage.write_object(copy, header, {'a': self.a})
                with storage.read_object(copy) as f:
                    self.assertEqual(f.header, header)
                    self.assertEqual(f.header.comments, header.comments)
                    self.assertEqual(f.header['COMMENT'], ['a comment card'])

    def test_complex_arrays(self):
        '''Complex data takes the complex type of the precision'''
        c = (self.a + 1j * self.a).astype(np.complex128)
        for data_format in ('npz', 'npy', 'hdf5'):
            with self.subTest(data_format=data_format):
                path = self._path('complex', data_format)
                storage.write_object(path, self.header, {'c': c})
                with storage.read_object(path) as f:
                    c64 = f.load('c', 1, target_device_idx=-1, precision=1)
                    c128 = f.load('c', 1, target_device_idx=-1, precision=0)
                    self.assertEqual((c64.dtype, c128.dtype), (np.complex64, np.complex128))
                    np.testing.assert_array_equal(c128, c)

    def test_reader_base_class(self):
        reader = storage.ObjectReader('path')
        with self.assertRaises(NotImplementedError):
            reader.has('a')
        with self.assertRaises(NotImplementedError):
            reader.read('a')


class TestRestore(unittest.TestCase):
    '''Checks made by restore() on the files it reads, and optional contents'''

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, name, header, arrays=None):
        path = os.path.join(self.tmp.name, name + '.npz')
        storage.write_object(path, header, arrays or {'a': np.zeros(2)})
        return path

    def test_wrong_object_type_or_version(self):
        from specula.data_objects.convolution_kernel import ConvolutionKernel
        from specula.data_objects.gaussian_convolution_kernel import GaussianConvolutionKernel
        from specula.data_objects.electric_field import ElectricField
        from specula.data_objects.layer import Layer
        from specula.data_objects.time_history import TimeHistory
        cases = [(ConvolutionKernel, {'VERSION': 1.0}, 'version'),
                 (GaussianConvolutionKernel, {'VERSION': 1.0}, 'version'),
                 (ElectricField, {'OBJ_TYPE': 'Layer'}, 'ElectricField'),
                 (Layer, {'OBJ_TYPE': 'ElectricField'}, 'Layer'),
                 (TimeHistory, {'VERSION': 2}, 'version')]
        for klass, header, message in cases:
            with self.subTest(klass=klass.__name__):
                path = self._write(klass.__name__, header)
                with self.assertRaisesRegex(ValueError, message):
                    klass.restore(path, target_device_idx=-1)

    def test_intmat_with_slope_statistics(self):
        from specula.data_objects.intmat import Intmat
        intmat = Intmat(np.arange(6, dtype=np.float32).reshape(3, 2),
                        slope_mm=np.array([[-1.0, 1.0]] * 3), slope_rms=np.arange(3.0),
                        pupdata_tag='pup', subapdata_tag='subap', norm_factor=2.0,
                        target_device_idx=-1)
        path = os.path.join(self.tmp.name, 'im.npz')
        intmat.save(path)
        restored = Intmat.restore(path, target_device_idx=-1)
        np.testing.assert_array_equal(cpuArray(restored.intmat), cpuArray(intmat.intmat))
        np.testing.assert_array_equal(restored.slope_mm, intmat.slope_mm)
        np.testing.assert_array_equal(restored.slope_rms, intmat.slope_rms)
        self.assertEqual((restored.pupdata_tag, restored.subapdata_tag, restored.norm_factor),
                         ('pup', 'subap', 2.0))

    def test_gaussian_kernel(self):
        '''Restore into a new object, or into an existing one with the same parameters'''
        from specula.data_objects.convolution_kernel import ConvolutionKernel
        from specula.data_objects.gaussian_convolution_kernel import GaussianConvolutionKernel

        def make(dimx=4, spot_size=1.0):
            return GaussianConvolutionKernel(dimx=dimx, dimy=4, pxscale=0.2, dimension=16,
                                             spot_size=spot_size, pupil_size_m=8.0,
                                             data_dir=self.tmp.name, target_device_idx=-1)
        kernel = make()
        kernel.build()
        kernel.calculate_lgs_map()
        kernel.save(os.path.join(self.tmp.name, 'kernel.fits'))
        files = [p for p in Path(self.tmp.name).iterdir() if storage.object_format(p)]
        self.assertEqual(len(files), 1)
        restored = GaussianConvolutionKernel.restore(files[0], target_device_idx=-1)
        self.assertEqual((restored.dimx, restored.spot_size), (4, 1.0))
        self.assertEqual(restored.real_kernels.shape[-2:], (16, 16))
        with self.assertRaisesRegex(ValueError, 'spot size'):
            GaussianConvolutionKernel.restore(files[0], kernel_obj=make(spot_size=2.0))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            GaussianConvolutionKernel.restore(files[0], kernel_obj=make(dimx=2))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            ConvolutionKernel.restore(files[0], kernel_obj=make(dimx=2))


class TestCalibrationFiles(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cm = CalibManager(self.tmp.name)
        os.makedirs(os.path.join(self.tmp.name, 'rec'))
        self.recmat = Recmat(np.arange(6, dtype=np.float32).reshape(3, 2), target_device_idx=-1)

    def test_extension_of_new_files(self):
        with patch.dict(os.environ, _env_without_format(), clear=True):
            self.assertTrue(self.cm.filename('Recmat', 'r1').endswith('.fits'))
            os.environ['SPECULA_DATA_FORMAT'] = 'npz'
            self.assertTrue(self.cm.filename('Recmat', 'r1').endswith('.npz'))
            # pickle is only for DataStore: calibration files stay FITS
            os.environ['SPECULA_DATA_FORMAT'] = 'pickle'
            self.assertTrue(self.cm.filename('Recmat', 'r1').endswith('.fits'))
        # An explicit extension is kept
        self.assertTrue(self.cm.filename('Recmat', 'r1.h5').endswith('r1.h5'))

    def test_existing_file_found_in_any_format(self):
        self.recmat.save(os.path.join(self.tmp.name, 'rec', 'r1.npz'))
        with patch.dict(os.environ, _env_without_format(), clear=True):
            self.assertTrue(self.cm.filename('Recmat', 'r1').endswith('r1.npz'))
            # With both, the file of the default format wins
            self.recmat.save(os.path.join(self.tmp.name, 'rec', 'r1.fits'))
            self.assertTrue(self.cm.filename('Recmat', 'r1').endswith('r1.fits'))
            os.environ['ITERAS_DATA_FORMAT'] = 'npz'
            self.assertTrue(self.cm.filename('Recmat', 'r1').endswith('r1.npz'))

    def test_simul_restores_tag_from_npz(self):
        from iteras.simul import Simul
        self.recmat.save(os.path.join(self.tmp.name, 'rec', 'r1.npz'))
        simul = Simul('dummy.yml')
        simul.cm = self.cm
        restored = simul.restore_object(Recmat, 'r1', target_device_idx=-1)
        np.testing.assert_array_equal(cpuArray(restored.recmat), cpuArray(self.recmat.recmat))

    def test_raw_data_files(self):
        os.makedirs(os.path.join(self.tmp.name, 'data'))
        data = np.arange(4.0)
        with patch.dict(os.environ, _env_without_format(), clear=True):
            os.environ['ITERAS_DATA_FORMAT'] = 'npz'
            self.cm.write_data('d1', data)
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, 'data', 'd1.npz')))
        np.testing.assert_array_equal(self.cm.read_data('d1'), data)


if __name__ == '__main__':
    unittest.main()
