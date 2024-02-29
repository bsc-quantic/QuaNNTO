import unittest
import numpy as np
from quannto.preprocessing import *

class TestPreprocessing(unittest.TestCase):
    def test_get_range(self):
        data = np.array([0, 1, 1.5, -2, 5, -3])
        expected_range = (-3, 5)
        obtained_range = get_range(data)
        self.assertEqual(expected_range, obtained_range)
        
    def test_trigonometric_feature_expressivity(self):
        features = np.array([[-np.pi], [-np.pi/2], [0], [np.pi/2], [np.pi]])
        num_final_features = 4
        expected_feats = np.array([[0, 0, 0, 0],
                                   [-1, 0, 3, 0],
                                   [0, 0, 0, 0],
                                   [1, 0, -3, 0],
                                   [0, 0, 0, 0]])
        obtained_feats = trigonometric_feature_expressivity(features, num_final_features)
        self.assertTrue(np.allclose(expected_feats, obtained_feats))
    
    def test_polynomial_feature_expressivity(self):
        features = np.array([[0.5], [1.0], [2.0]])
        num_final_features = 3
        expected_feats = np.array([[0.5, 2*0.5**2, 3*0.5**3],
                                   [1.0, 2*1.0**2, 3*1.0**3],
                                   [2.0, 2*2.0**2, 3*2.0**3]], dtype=float)
        obtained_feats = polynomial_feature_expressivity(features, num_final_features)
        self.assertTrue(np.allclose(expected_feats, obtained_feats))
        
    def test_rescale_data(self):
        data = np.array([0, 1, 1.5, -2, 5, -3])
        data_range = (-3, 5)
        scale_data_range = (0, 1)
        expected_rescaling = np.array([0.375, 0.5, 0.5625, 0.125, 1, 0])
        obtained_rescaling = rescale_data(data, data_range, scale_data_range)
        self.assertTrue(np.allclose(expected_rescaling, obtained_rescaling))
        
    def test_rescale_set(self):
        set = np.array([[0,1,2,3,4],
                        [4,0,1,2,3],
                        [3,4,0,1,2],
                        [2,3,4,0,1],
                        [1,2,3,4,0]], dtype=float)
        rescale_range = (0,1)
        expected_rescaled_set = np.array([[0, 0.25, 0.5, 0.75, 1],
                                          [1, 0, 0.25, 0.5, 0.75],
                                          [0.75, 1, 0, 0.25, 0.5],
                                          [0.5, 0.75, 1, 0, 0.25],
                                          [0.25, 0.5, 0.75, 1, 0]])
        obtained_rescaled_set = rescale_set(set, rescale_range)
        self.assertTrue(np.allclose(expected_rescaled_set, obtained_rescaled_set))
        
    def test_binning(self):
        multidata = np.array([0, 0.5, 0.99, 1.5, 2.3, 3.01, 3.9])
        range = (0,4)
        num_cats = 4
        expected_binnings = np.array([0, 0, 0, 1, 2, 3, 3])
        obtained_binnings = np.array([binning(data, range, num_cats) for data in multidata])
        self.assertTrue(np.allclose(obtained_binnings, expected_binnings))