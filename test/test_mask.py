#!/usr/bin/env python
#%%
import specula
specula.init(-1, precision=1)

from numpy import random
import unittest
import numpy as np

from specula.lib.mask import BaseMask, CircularMask, AnnularMask


class BaseMaskTest(unittest.TestCase):

    def test_comparison(self):
        mask1 = random.choice([True, False], size=(3, 4))
        mask2 = mask1.copy()
        self.assertTrue(id(mask1) != id(mask2))
        self.assertTrue(BaseMask(mask1) == BaseMask(mask2))

    def test_comparison_different_shape(self):
        mask = random.choice([True, False], size=(3, 4))
        mask1 = BaseMask(mask)
        mask2 = BaseMask(mask.reshape(1, 12))
        self.assertTrue(mask1 != mask2)

    def test_hash(self):
        mask1 = random.choice([True, False], size=(3, 4))
        mask2 = mask1.copy()
        bm1 = BaseMask(mask1)
        bm2 = BaseMask(mask2)
        self.assertTrue(hash(bm1) == hash(bm2))


class MaskTest(unittest.TestCase):

    def testStandard(self):
        mask= CircularMask((13, 14))
        self.assertEqual(mask.shape(), (13, 14))

    def testNullRadiusMakeNullMask(self):
        mask= CircularMask((10, 20), maskRadius=0)
        self.assertTrue(np.all(mask.mask()))

    def testPassingRadiusNone(self):
        mask= CircularMask((10, 10))
        self.assertEqual(5, mask.radius())

    def testPassingCenterNone(self):
        mask= CircularMask((20, 10))
        self.assertTrue(np.allclose([10, 5],
                                    mask.center()),
                        "center is %s" % mask.center())

    def testRegionOfInterest(self):
        shape= (140, 100)
        aMask= CircularMask(shape, maskRadius=20, maskCenter=(60, 45))
        with self.assertRaises(NotImplementedError):
            roi= aMask.regionOfInterest()

    def testAsTransmissionValue(self):
        mask= CircularMask((10, 10))
        transmission= mask.asTransmissionValue()
        self.assertEqual(1, transmission[5, 5])
        self.assertEqual(0, transmission[0, 0])
        
    def testAnularMask(self):
        mask1= CircularMask((10, 10),4,(5,5))
        mask2= CircularMask((10, 10),2,(5,5))
        mask3 = AnnularMask ((10, 10),4,(5,5),2)
        mask4 = mask1.mask() | ~mask2.mask()
        self.assertEqual(mask3.mask().all(), mask4.all())
               
