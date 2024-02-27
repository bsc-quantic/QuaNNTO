import unittest
import numpy as np
from quannto.expectation_value import *

class TestExpectationValue(unittest.TestCase):
    def test_compute_K_exp_vals(self):
        V = np.array([[2.5, 0, 2, 0], 
                      [0, 2.5, 0, -2],
                      [2, 0, 2, 0],
                      [0, -2, 0, 2]])
        expected_K_exp_vals = np.array([[[0.125+1.j, 0.   +0.j], 
                                         [0.   +0.j, 0.125-1.j]], 
                                        [[0.625+0.j, 0.   +0.j], 
                                         [0.   +0.j, 0.625+0.j]], 
                                        [[1.625+0.j, 0.   +0.j], 
                                         [0.   +0.j, 1.625+0.j]], 
                                        [[0.125-1.j, 0.   +0.j], 
                                         [0.   +0.j, 0.125+1.j]]])
        obtained_K_exp_vals = compute_K_exp_vals(V)
        
        self.assertTrue(np.allclose(expected_K_exp_vals, obtained_K_exp_vals))
        
    def test_ladder_ops_trace_expression(self):
        modes_expr, types_expr = ladder_ops_trace_expression(2, 1)
        expected_modes_expr = [[[0, 1], [0, 0], [0, 1], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [0, 0], [0, 1], [0, 1], [0, 0], [1, 0], [1, 1], [1, 0], [1, 1]]]
        expected_types_expr = [[[0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 0], [1, 0], [1, 1], [1, 1]]]
        
        self.assertTrue(np.allclose(np.array(expected_types_expr), np.array(types_expr)))
        self.assertTrue(np.allclose(np.array(expected_modes_expr), np.array(modes_expr)))
    
    def test_to_ladder_expression(self):
        symbolic_expr = "c0c1c2a0a1a2"
        expected_modes = [0,1,2,0,1,2]
        expected_types = [1,1,1,0,0,0]
        
        obtained_modes, obtained_types = to_ladder_expression(symbolic_expr)
        
        self.assertTrue(np.allclose(np.array(expected_modes), np.array(obtained_modes)))
        self.assertTrue(np.allclose(np.array(expected_types), np.array(obtained_types)))
        
    def test_include_observable(self):
        ladder_modes = [[0, 1], [0, 0], [0, 1], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [0, 0], [0, 1], [0, 1], [0, 0], [1, 0], [1, 1], [1, 0], [1, 1]]
        ladder_types = [[0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 0], [1, 0], [1, 1], [1, 1]]
        obs_modes = [[0,0], [1,1]]
        obs_types = [[1,0], [1,0]]
        
        expected_obs_modes = [[[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 1]], [[0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]]]
        expected_obs_types = [[[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 0, 1]], [[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 0, 1]]]
        obt_obs_modes, obt_obs_types = include_observable(ladder_modes, ladder_types, obs_modes, obs_types)
        
        self.assertTrue(np.allclose(np.array(expected_obs_modes), np.array(obt_obs_modes)))
        self.assertTrue(np.allclose(np.array(expected_obs_types), np.array(obt_obs_types)))

    def test_perfect_matching(self):
        obtained_perf_matchings = perfect_matchings(4)
        expected_perf_matchings = [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]]
        
        self.assertTrue(np.allclose(np.array(expected_perf_matchings), np.array(obtained_perf_matchings)))
        
    def test_ladder_exp_val(self):
        perf_matchings = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]])
        ladder_modes = np.array([0, 0, 0, 0])
        ladder_types = np.array([0, 1, 0, 1])
        K_exp_vals = np.array([[[0.125+1.j, 0.   +0.j], 
                                [0.   +0.j, 0.125-1.j]], 
                               [[0.625+0.j, 0.   +0.j], 
                                [0.   +0.j, 0.625+0.j]], 
                               [[1.625+0.j, 0.   +0.j], 
                                [0.   +0.j, 1.625+0.j]], 
                               [[0.125-1.j, 0.   +0.j], 
                                [0.   +0.j, 0.125+1.j]]])
        expected_exp_val = 4.671875+0j
        obtained_exp_val = ladder_exp_val(perf_matchings, ladder_modes, ladder_types, K_exp_vals)
        
        self.assertEqual(expected_exp_val, obtained_exp_val)
        
    def test_get_symplectic_coefs(self):
        N = 2
        ladder_modes = np.array([[[0, 1], [0, 0], [0, 1], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [0, 0], [0, 1], [0, 1], [0, 0], [1, 0], [1, 1], [1, 0], [1, 1]]])
        ladder_types = np.array([[[0, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 1], [0, 1], [0, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 0], [1, 0], [1, 1], [1, 1]]])
        S = np.array([[[0.1, 0.2, 0.3, 0.4],
                       [1. , 1. , 1. , 1. ],
                       [1. , 1. , 1. , 1. ],
                       [1. , 1. , 1. , 1. ]]])
        expected_symp_coefs = np.array([[0.04, 0.01, 0.02, 0.03, 0.06, 0.02, 0.04, 0.08, 0.09, 0.12, 0.06, 0.03, 0.12, 0.16, 0.04, 0.08]])
        obtained_symp_coefs = get_symplectic_coefs(N, S, ladder_modes, ladder_types)
        
        self.assertTrue(np.allclose(np.array(expected_symp_coefs), np.array(obtained_symp_coefs)))
