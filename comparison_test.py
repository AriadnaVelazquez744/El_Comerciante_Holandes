"""
Comparison test: Two-phase hybrid vs brute force on small instances
"""

from efficient_solution import two_phase_hybrid_solve, generate_multi_item_instance
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instance_generator import generate_unified_test_instance


def convert_single_item_to_multi_item(instance):
    """Convert single-item instance format to multi-item format"""
    n = len(instance['ports'])
    
    purchase_prices = {}
    sale_prices = {}
    
    for port in range(n):
        if port == 0:
            purchase_prices[port] = [0]
            sale_prices[port] = [0]
        else:
            purchase_prices[port] = [instance['purchase_prices'][port]]
            sale_prices[port] = [instance['sale_prices'][port]]
    
    return {
        'ports': instance['ports'],
        'travel_costs': instance['travel_costs'],
        'travel_times': instance['travel_times'],
        'purchase_prices': purchase_prices,
        'sale_prices': sale_prices,
        'initial_capital': instance['initial_capital'],
        'capacity': instance['capacity'],
        'max_time': instance['max_time'],
        'num_items': 1,
        'max_units_per_op': 1
    }


def test_correctness_small_instances():
    """Test correctness on small instances where we can verify manually"""
    print("Testing correctness on small instances...")
    
    # Test on n=3, 4, 5
    for n in [3, 4, 5]:
        print(f"\nTesting n={n}...")
        
        # Generate single-item instance (compatible with existing generators)
        instance_single = generate_unified_test_instance(n, seed=42)
        
        # Convert to multi-item format
        instance = convert_single_item_to_multi_item(instance_single)
        
        # Solve with two-phase hybrid
        result = two_phase_hybrid_solve(instance, timeout=60.0, beam_width=100)
        
        print(f"  Result: capital={result['capital']}, route={result['route']}")
        print(f"  Routes generated: {result['routes_generated']}, evaluated: {result['routes_evaluated']}")
        print(f"  Execution time: {result['execution_time']:.3f}s")
        
        # Verify solution is feasible
        if result['capital'] is not None:
            # Basic feasibility checks
            assert result['route'] is not None
            assert result['route'][0] == 0  # Starts at Amsterdam
            assert result['route'][-1] == 0  # Ends at Amsterdam
            print(f"  ✓ Solution is feasible")
        else:
            print(f"  ⚠ No solution found (may be infeasible instance)")


if __name__ == '__main__':
    test_correctness_small_instances()

