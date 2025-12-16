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

    def _from_masked_array_test(self, orig_mask, retrieved_mask):
#        self._logger.debug("rays %5.2f %5.2f " %
#                           (orig_mask.radius(), retrieved_mask.radius()))
#        self._logger.debug("center x %5.2f %5.2f " %
#                           (orig_mask.center()[0], retrieved_mask.center()[0]))
#        self._logger.debug("center y %5.2f %5.2f " %
#                           (orig_mask.center()[1], retrieved_mask.center()[1]))
#        self._logger.debug("number of masked pix: %d  vs %d " % (
#            retrieved_mask.in_mask_indices().size, orig_mask.in_mask_indices().size))
        np.testing.assert_allclose(
            orig_mask.radius(), retrieved_mask.radius(), rtol=0.1)
        np.testing.assert_allclose(
            orig_mask.center(), retrieved_mask.center(), atol=0.5)
        np.testing.assert_equal(orig_mask.shape(), retrieved_mask.shape())

    # @unittest.skip('Masked array not working')
    # def testCreateCircularMaskFromMaskedArrayWithCircleParametersEstimationCOG(self):
    #     shape= (140, 100)
    #     aMask= CircularMask(shape, maskRadius=20, maskCenter=(60, 45))
    #     maskedArray= np.ma.masked_array(np.ones(shape), mask=aMask.mask())
    #     retrievedMask= CircularMask.fromMaskedArray(maskedArray,method='COG')
    #     self._from_masked_array_test(aMask, retrievedMask)

    # @unittest.skip('Masked array not working')
    # def testCreateCircularMaskFromMaskedArrayWithCircleParametersEstimationImageMoments(self):
    #     shape= (140, 100)
    #     aMask= CircularMask(shape, maskRadius=20, maskCenter=(60, 45))
    #     maskedArray= np.ma.masked_array(np.ones(shape), mask=aMask.mask())
    #     retrievedMask= CircularMask.fromMaskedArray(maskedArray,method='ImageMoments')
    #     self._from_masked_array_test(aMask, retrievedMask)
    #     self.assertTrue(np.isin(retrievedMask.in_mask_indices(),
    #                     aMask.in_mask_indices()).all())

    # @unittest.skip('Masked array not working')
    # def testCreateCircularMaskFromMaskedArrayWithCircleParametersEstimationCorrelation(self):
    #     shape= (140, 100)
    #     aMask= CircularMask(shape, maskRadius=20, maskCenter=(60, 45))
    #     maskedArray= np.ma.masked_array(np.ones(shape), mask=aMask.mask())
    #     retrievedMask= CircularMask.fromMaskedArray(maskedArray,method='correlation')
    #     self._from_masked_array_test(aMask, retrievedMask)

    # @unittest.skip('Masked array not working')
    # def testCreateCircularMaskFromMaskedArrayWithCircleParametersEstimationRANSAC(self):
    #     shape= (140, 100)
    #     aMask= CircularMask(shape, maskRadius=20, maskCenter=(60, 45))
    #     maskedArray= np.ma.masked_array(np.ones(shape), mask=aMask.mask())
    #     retrievedMask= CircularMask.fromMaskedArray(maskedArray,method='RANSAC')
    #     self._from_masked_array_test(aMask, retrievedMask)

    # @unittest.skip('Masked array not working')
    # def testCreateCircularMaskFromMaskedArrayWithAReallyCircularMask(self):
    #     shape= (140, 100)
    #     aMask= CircularMask(shape, maskRadius=20, maskCenter=(60, 40))
    #     maskedArray= np.ma.masked_array(np.ones(shape), mask=aMask.mask())
    #     retrievedMask= CircularMask.fromMaskedArray(maskedArray)
    #     self._from_masked_array_test(aMask, retrievedMask)
    #     self.assertTrue(np.isin(retrievedMask.in_mask_indices(),
    #                     aMask.in_mask_indices()).all())

    # @unittest.skip('Masked array not working')
    # def testCreateCircularMaskFromMaskedArrayWithFloats(self):
    #     aMask = CircularMask((486, 640), 126.32, (235.419, 309.468))
    #     marray = np.ma.array(data=np.ones((486, 640)), mask=aMask.mask())
        
    #     retrievedMask = CircularMask.fromMaskedArray(marray)
    #     self._from_masked_array_test(aMask, retrievedMask)
    #     self.assertTrue(np.isin(retrievedMask.in_mask_indices(),
    #                     aMask.in_mask_indices()).all())

    # @unittest.skip('Masked array not working')
    # def testCreateCircularMaskFromMaskedArrayWithNonCircularMask(self):
    #     shape= (140, 100)
    #     aSquareMask= np.ones((shape))
    #     aSquareMask[40:80, 20:60]= 0
    #     maskedArray= np.ma.masked_array(np.ones(shape),
    #                                     mask=aSquareMask.astype(bool))
    #     retrievedMask= CircularMask.fromMaskedArray(maskedArray)
    #     self.assertTrue(
    #         np.in1d(retrievedMask.in_mask_indices(),
    #                 np.argwhere(aSquareMask.flatten() == False)).all())

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
               
