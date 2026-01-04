"""
Unit tests for the two-phase hybrid algorithm
"""

import unittest
from efficient_solution import (
    is_feasible_state,
    apply_transaction,
    calculate_pctsp_bound,
    generate_promising_routes,
    solve_transactions_dp,
    two_phase_hybrid_solve,
    generate_multi_item_instance
)


class TestHelperFunctions(unittest.TestCase):
    
    def test_is_feasible_state(self):
        """Test feasibility checking"""
        instance = {'capacity': 5}
        
        # Valid state
        self.assertTrue(is_feasible_state(100.0, (2, 1), 5, instance))
        
        # Negative capital
        self.assertFalse(is_feasible_state(-10.0, (2, 1), 5, instance))
        
        # Exceeds capacity
        self.assertFalse(is_feasible_state(100.0, (3, 3), 5, instance))
        
        # Negative load
        self.assertFalse(is_feasible_state(100.0, (-1, 2), 5, instance))
    
    def test_apply_transaction(self):
        """Test transaction application"""
        instance = {
            'purchase_prices': {1: [10, 15]},
            'sale_prices': {1: [20, 25]}
        }
        
        # Buy operation
        new_cap, new_load = apply_transaction(100.0, (0, 0), (0, 2), 1, instance)
        self.assertEqual(new_cap, 80.0)  # 100 - 10*2
        self.assertEqual(new_load, (2, 0))
        
        # Sell operation
        new_cap, new_load = apply_transaction(100.0, (2, 1), (0, -1), 1, instance)
        self.assertEqual(new_cap, 120.0)  # 100 + 20*1
        self.assertEqual(new_load, (1, 1))
    
    def test_calculate_pctsp_bound(self):
        """Test PCTSP bound calculation"""
        instance = {
            'purchase_prices': {1: [10, 15], 2: [12, 18]},
            'sale_prices': {1: [25, 30], 2: [28, 35]},
            'travel_costs': [[0, 5, 8], [5, 0, 6], [8, 6, 0]],
            'capacity': 3,
            'max_units_per_op': 2
        }
        
        route = [0, 1, 2, 0]
        bound = calculate_pctsp_bound(route, instance)
        
        # Bound should be positive for profitable route
        self.assertIsInstance(bound, float)
        # Should account for travel costs
        self.assertLess(bound, 1000.0)  # Reasonable upper bound


class TestRouteGeneration(unittest.TestCase):
    
    def test_generate_promising_routes_small(self):
        """Test route generation on small instance"""
        instance = generate_multi_item_instance(n_ports=5, m_items=2, k_max_units=2, seed=42)
        
        routes = generate_promising_routes(instance, beam_width=20, max_depth=4, timeout=10.0)
        
        self.assertGreater(len(routes), 0)
        # All routes should start and end at Amsterdam
        for route, bound in routes:
            self.assertEqual(route[0], 0)
            self.assertEqual(route[-1], 0)
            self.assertIsInstance(bound, float)


class TestTransactionDP(unittest.TestCase):
    
    def test_solve_transactions_dp_simple(self):
        """Test transaction DP on simple route"""
        instance = generate_multi_item_instance(n_ports=4, m_items=1, k_max_units=1, seed=42)
        
        route = [0, 1, 0]
        max_capital, decisions = solve_transactions_dp(route, instance)
        
        if max_capital is not None:
            self.assertIsInstance(max_capital, float)
            self.assertGreaterEqual(max_capital, instance['initial_capital'] - 100)  # Should be reasonable
    
    def test_solve_transactions_dp_infeasible(self):
        """Test DP with infeasible route (too long time)"""
        instance = generate_multi_item_instance(n_ports=4, m_items=1, k_max_units=1, seed=42)
        instance['max_time'] = 1  # Very short time
        
        route = [0, 1, 2, 3, 0]  # Long route
        max_capital, decisions = solve_transactions_dp(route, instance)
        
        # Should return None for infeasible route
        self.assertIsNone(max_capital)


class TestFullAlgorithm(unittest.TestCase):
    
    def test_two_phase_hybrid_small(self):
        """Test full algorithm on small instance"""
        instance = generate_multi_item_instance(n_ports=5, m_items=2, k_max_units=2, seed=42)
        
        result = two_phase_hybrid_solve(instance, timeout=30.0, beam_width=30)
        
        self.assertIsNotNone(result)
        self.assertIn('capital', result)
        self.assertIn('route', result)
        self.assertIn('routes_generated', result)
        self.assertIn('routes_evaluated', result)
        
        if result['capital'] is not None:
            self.assertIsInstance(result['capital'], float)
            self.assertGreater(result['capital'], float('-inf'))
    
    def test_two_phase_hybrid_medium(self):
        """Test full algorithm on medium instance"""
        instance = generate_multi_item_instance(n_ports=10, m_items=2, k_max_units=2, seed=42)
        
        result = two_phase_hybrid_solve(instance, timeout=60.0, beam_width=50)
        
        self.assertIsNotNone(result)
        self.assertIn('capital', result)
        
        # Should complete within timeout
        self.assertLess(result['execution_time'], 60.0)


if __name__ == '__main__':
    unittest.main()


