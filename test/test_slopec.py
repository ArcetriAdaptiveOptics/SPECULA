import specula
specula.init(0)  # Default target device

import unittest

from specula import np

from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from specula.processing_objects.slopec import Slopec
from test.specula_testlib import cpu_and_gpu

NSUBAPS = 4
NSLOPES = NSUBAPS * 2
NMODES = 2


class _TestSlopec(Slopec):
    """Minimal Slopec subclass, just enough to run Slopec.__init__()"""
    def nsubaps(self):
        return NSUBAPS

    def nslopes(self):
        return NSLOPES


class TestSlopec(unittest.TestCase):

    @cpu_and_gpu
    def test_filt_matrices_checks(self, target_device_idx, xp):
        """
        Test the consistency checks on filt_intmat, filt_recmat and filtmat.
        """
        filt_intmat = Intmat(np.zeros((NSLOPES, NMODES)), target_device_idx=target_device_idx)
        filt_recmat = Recmat(np.zeros((NMODES, NSLOPES)), target_device_idx=target_device_idx)
        filtmat = [np.zeros((NSLOPES, NMODES)), np.zeros((NMODES, NSLOPES))]

        # Only one of filt_intmat / filt_recmat
        with self.assertRaisesRegex(ValueError, 'missing: filt_recmat'):
            _TestSlopec(filt_intmat=filt_intmat, target_device_idx=target_device_idx)
        with self.assertRaisesRegex(ValueError, 'missing: filt_intmat'):
            _TestSlopec(filt_recmat=filt_recmat, target_device_idx=target_device_idx)

        # filtmat together with filt_intmat / filt_recmat
        with self.assertRaisesRegex(ValueError, 'filt_intmat must not be set'):
            _TestSlopec(filtmat=filtmat, filt_intmat=filt_intmat,
                        target_device_idx=target_device_idx)
        with self.assertRaisesRegex(ValueError, 'filt_recmat must not be set'):
            _TestSlopec(filtmat=filtmat, filt_recmat=filt_recmat,
                        target_device_idx=target_device_idx)

        # Valid combinations do not raise
        slopec = _TestSlopec(target_device_idx=target_device_idx)
        self.assertIsNone(slopec.filt_intmat)
        self.assertIsNone(slopec.filt_recmat)

        slopec = _TestSlopec(filt_intmat=filt_intmat, filt_recmat=filt_recmat,
                             target_device_idx=target_device_idx)
        self.assertIs(slopec.filt_intmat, filt_intmat)
        self.assertIs(slopec.filt_recmat, filt_recmat)

        slopec = _TestSlopec(filtmat=filtmat, target_device_idx=target_device_idx)
        self.assertIsInstance(slopec.filt_intmat, Intmat)
        self.assertIsInstance(slopec.filt_recmat, Recmat)
