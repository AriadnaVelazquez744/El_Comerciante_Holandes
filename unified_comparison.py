"""
Unified comparison framework for the three brute force algorithms.
All algorithms share the same base structure for fair comparison.
"""
import itertools
import math
import time
from pure_brute_force import is_valid_tour


def unified_route_generator(visitable_ports, costs, travel_times, T_max, timeout, start_time):
    """
    Unified route generation for all algorithms.
    Yields (tour, min_time_estimate) for each valid route.
    """
    for k in range(0, len(visitable_ports) + 1):
        if time.time() - start_time > timeout:
            break
            
        for subset in itertools.combinations(visitable_ports, k):
            if time.time() - start_time > timeout:
                break

            # Quick time-based pruning
            if len(subset) > 0:
                min_time = 0
                current = 0
                remaining = list(subset)
                
                while remaining:
                    next_port = min(remaining, key=lambda p: travel_times[current][p])
                    min_time += travel_times[current][next_port] + 1
                    current = next_port
                    remaining.remove(next_port)
                
                min_time += travel_times[current][0]
                
                if min_time > T_max:
                    continue

            for permutation in itertools.permutations(subset):
                if time.time() - start_time > timeout:
                    break
                    
                tour = [0] + list(permutation) + [0]
                yield tour


def pure_brute_force_unified(instance, timeout=200.0):
    """
    Pure brute force with unified structure - minimal overhead baseline.
    """
    start_time = time.time()
    
    ports = instance['ports']
    n = len(ports)
    costs = instance['travel_costs']
    travel_times = instance['travel_times']
    purchase_prices = instance['purchase_prices']
    sale_prices = instance['sale_prices']
    r = instance['initial_capital']
    B = instance['capacity']
    T_max = instance['max_time']

    best_capital = -math.inf
    best_tour = None
    best_decisions = None
    best_time = 0
    explored_solutions = 0
    timeout_reached = False

    visitable_ports = list(range(1, n))

    for tour in unified_route_generator(visitable_ports, costs, travel_times, T_max, timeout, start_time):
        if timeout_reached:
            break
            
        num_decisions = len(tour) - 2

        for decisions in itertools.product([0, 1, 2], repeat=num_decisions):
            if explored_solutions % 1000 == 0 and time.time() - start_time > timeout:
                timeout_reached = True
                break
                
            explored_solutions += 1

            # Simulate tour
            capital = r
            load = 0
            total_time = 0
            feasible = True

            for i in range(len(tour) - 1):
                current_port = tour[i]
                next_port = tour[i + 1]

                capital -= costs[current_port][next_port]
                total_time += travel_times[current_port][next_port]

                if capital < 0 or total_time > T_max:
                    feasible = False
                    break

                if i < len(tour) - 2:
                    decision = decisions[i]

                    if decision == 1:  # BUY
                        if load >= B or capital < purchase_prices[next_port]:
                            feasible = False
                            break
                        capital -= purchase_prices[next_port]
                        load += 1
                    elif decision == 2:  # SELL
                        if load <= 0:
                            feasible = False
                            break
                        capital += sale_prices[next_port]
                        load -= 1

                    total_time += 1

                    if capital < 0 or total_time > T_max:
                        feasible = False
                        break

            if feasible and not timeout_reached:
                if is_valid_tour(tour, capital, r, total_time):
                    if capital > best_capital:
                        best_capital = capital
                        best_tour = tour.copy()
                        best_decisions = list(decisions)
                        best_time = total_time

    return {
        'optimal_tour': best_tour,
        'optimal_decisions': best_decisions,
        'final_capital': best_capital if best_capital > -math.inf else None,
        'total_time': best_time,
        'explored_solutions': explored_solutions,
        'timeout': timeout_reached
    }


def pruned_brute_force_unified(instance, timeout=200.0):
    """
    Pruned brute force with unified structure - same base as pure, with early termination.
    """
    start_time = time.time()
    
    ports = instance['ports']
    n = len(ports)
    costs = instance['travel_costs']
    travel_times = instance['travel_times']
    purchase_prices = instance['purchase_prices']
    sale_prices = instance['sale_prices']
    r = instance['initial_capital']
    B = instance['capacity']
    T_max = instance['max_time']

    best_capital = -math.inf
    best_tour = None
    best_decisions = None
    best_time = 0
    branches_explored = 0
    branches_pruned = 0
    complete_solutions = 0
    timeout_reached = False

    visitable_ports = list(range(1, n))

    for tour in unified_route_generator(visitable_ports, costs, travel_times, T_max, timeout, start_time):
        if timeout_reached:
            break
            
        num_decisions = len(tour) - 2

        for decisions in itertools.product([0, 1, 2], repeat=num_decisions):
            if branches_explored % 1000 == 0 and time.time() - start_time > timeout:
                timeout_reached = True
                break
                
            branches_explored += 1

            # Simulate tour with early termination
            capital = r
            load = 0
            total_time = 0
            pruned_early = False

            for i in range(len(tour) - 1):
                current_port = tour[i]
                next_port = tour[i + 1]

                capital -= costs[current_port][next_port]
                total_time += travel_times[current_port][next_port]

                # Early termination checks (the only difference from pure BF)
                if capital < 0:
                    branches_pruned += 1
                    pruned_early = True
                    break
                if total_time > T_max:
                    branches_pruned += 1
                    pruned_early = True
                    break

                if i < len(tour) - 2:
                    decision = decisions[i]

                    if decision == 1:  # BUY
                        if load >= B:
                            branches_pruned += 1
                            pruned_early = True
                            break
                        if capital < purchase_prices[next_port]:
                            branches_pruned += 1
                            pruned_early = True
                            break
                        capital -= purchase_prices[next_port]
                        load += 1
                    elif decision == 2:  # SELL
                        if load <= 0:
                            branches_pruned += 1
                            pruned_early = True
                            break
                        capital += sale_prices[next_port]
                        load -= 1

                    total_time += 1

                    # Early termination after operation
                    if capital < 0:
                        branches_pruned += 1
                        pruned_early = True
                        break
                    if total_time > T_max:
                        branches_pruned += 1
                        pruned_early = True
                        break

            if not pruned_early and not timeout_reached:
                complete_solutions += 1
                if is_valid_tour(tour, capital, r, total_time):
                    if capital > best_capital:
                        best_capital = capital
                        best_tour = tour.copy()
                        best_decisions = list(decisions)
                        best_time = total_time

    return {
        'optimal_tour': best_tour,
        'optimal_decisions': best_decisions,
        'final_capital': best_capital if best_capital > -math.inf else None,
        'total_time': best_time,
        'explored_solutions': branches_explored,
        'complete_solutions_evaluated': complete_solutions,
        'branches_pruned': branches_pruned,
        'timeout': timeout_reached
    }


def hybrid_dp_unified(instance, timeout=200.0):
    """
    Hybrid DP with unified structure - same route generation, DP for decisions.
    """
    start_time = time.time()
    
    ports = instance['ports']
    n = len(ports)
    costs = instance['travel_costs']
    travel_times = instance['travel_times']
    purchase_prices = instance['purchase_prices']
    sale_prices = instance['sale_prices']
    r = instance['initial_capital']
    B = instance['capacity']
    T_max = instance['max_time']

    best_capital = -math.inf
    best_tour = None
    best_decisions = None
    best_time = 0
    routes_explored = 0
    dp_executions = 0
    timeout_reached = False

    # Estimate capital range for DP discretization
    # Use finer discretization to ensure accuracy (but still much better than 3^n)
    max_capital = r + B * max(sale_prices)
    # Use step of 1 for accuracy (can be optimized later)
    discretization_step = 1
    num_capital_levels = int(max_capital) + 10  # Small buffer

    def dp_for_route(route):
        """Optimized DP for a fixed route."""
        nonlocal dp_executions
        dp_executions += 1
        
        L = len(route) - 2
        if L == 0:
            return None, None

        # DP table: dp[capital_level][load] = max_real_capital
        dp_prev = [[-math.inf] * (B + 1) for _ in range(num_capital_levels)]
        dp_curr = [[-math.inf] * (B + 1) for _ in range(num_capital_levels)]
        
        # Initialize: Start at Amsterdam (port 0)
        # First, travel to first port in route
        first_port = route[1]
        travel_cost_to_first = costs[0][first_port]
        capital_after_first_travel = r - travel_cost_to_first
        
        if capital_after_first_travel < 0:
            return None, None  # Can't even reach first port
        
        init_cap_level = min(int(capital_after_first_travel), num_capital_levels - 1)
        if init_cap_level < 0:
            return None, None
        dp_prev[init_cap_level][0] = capital_after_first_travel

        # Track decisions for reconstruction
        decisions_track = [[[None] * (B + 1) for _ in range(num_capital_levels)] for _ in range(L + 1)]

        for i in range(L):
            port = route[i + 1]  # Current port we're at (already traveled here)
            
            # Clear current
            for cap in range(num_capital_levels):
                for load in range(B + 1):
                    dp_curr[cap][load] = -math.inf

            # Process each state - dp_prev represents state AFTER arriving at current port
            for cap_level in range(num_capital_levels):
                for load in range(B + 1):
                    capital = dp_prev[cap_level][load]
                    if capital < 0:
                        continue

                    # We're at port route[i+1] with capital and load
                    # First, make a decision (buy/sell/nothing)
                    # Then, travel to next port
                    
                    # Decision options at current port
                    decision_options = []
                    
                    # Option 0: Do nothing
                    decision_options.append((0, capital, load))
                    
                    # Option 1: Buy
                    if load < B and capital >= purchase_prices[port]:
                        decision_options.append((1, capital - purchase_prices[port], load + 1))
                    
                    # Option 2: Sell
                    if load > 0:
                        decision_options.append((2, capital + sale_prices[port], load - 1))
                    
                    # After decision, travel to next port
                    if i < L - 1:
                        next_port = route[i + 2]
                        travel_cost = costs[port][next_port]
                    else:
                        travel_cost = costs[port][0]  # Return to Amsterdam
                    
                    # Apply travel cost and update DP
                    for dec, cap_after_decision, load_after_decision in decision_options:
                        cap_after_travel = cap_after_decision - travel_cost
                        
                        if cap_after_travel < 0:
                            continue  # Infeasible
                            
                        new_cap_level = min(int(cap_after_travel), num_capital_levels - 1)
                        if new_cap_level < 0:
                            continue
                        
                        # Update DP: state after arriving at next port
                        if cap_after_travel > dp_curr[new_cap_level][load_after_decision]:
                            dp_curr[new_cap_level][load_after_decision] = cap_after_travel
                            decisions_track[i + 1][new_cap_level][load_after_decision] = (dec, load, cap_level)


            dp_prev, dp_curr = dp_curr, dp_prev

        # Find best final state
        best_final = -math.inf
        best_load = 0
        best_cap_level = 0
        
        for cap_level in range(num_capital_levels):
            for load in range(B + 1):
                if dp_prev[cap_level][load] > best_final:
                    best_final = dp_prev[cap_level][load]
                    best_load = load
                    best_cap_level = cap_level

        if best_final < 0:
            return None, None

        # Reconstruct decisions
        decisions = []
        curr_load = best_load
        curr_cap = best_cap_level
        
        for i in range(L, 0, -1):
            dec_info = decisions_track[i][curr_cap][curr_load]
            if dec_info is None:
                return None, None
            dec, prev_load, prev_cap = dec_info
            decisions.append(dec)
            curr_load = prev_load
            curr_cap = prev_cap

        decisions.reverse()
        return best_final, decisions

    visitable_ports = list(range(1, n))

    for tour in unified_route_generator(visitable_ports, costs, travel_times, T_max, timeout, start_time):
        if timeout_reached:
            break
            
        routes_explored += 1
        
        # Use DP instead of enumerating decisions
        final_capital, decisions = dp_for_route(tour)
        
        if final_capital is not None:
            # Calculate total time
            total_time = 0
            for i in range(len(tour) - 1):
                total_time += travel_times[tour[i]][tour[i + 1]]
            for dec in decisions:
                if dec != 0:
                    total_time += 1

            if is_valid_tour(tour, final_capital, r, total_time):
                if final_capital > best_capital:
                    best_capital = final_capital
                    best_tour = tour.copy()
                    best_decisions = decisions.copy()
                    best_time = total_time

    return {
        'optimal_tour': best_tour,
        'optimal_decisions': best_decisions,
        'final_capital': best_capital if best_capital > -math.inf else None,
        'total_time': best_time,
        'routes_explored': routes_explored,
        'dp_executions': dp_executions,
        'timeout': timeout_reached
    }


def compare_all_unified(instance, timeout=200.0):
    """
    Compare all three unified algorithms on the same instance.
    """
    print("="*80)
    print("UNIFIED COMPARISON - All algorithms share the same base structure")
    print("="*80)
    
    results = {}
    
    # Pure BF
    print("\n1. Pure Brute Force (baseline)...")
    start = time.time()
    result_pure = pure_brute_force_unified(instance, timeout)
    time_pure = time.time() - start
    results['pure'] = (result_pure, time_pure)
    print(f"   Time: {time_pure:.4f}s, Explored: {result_pure.get('explored_solutions', 'N/A')}")
    
    # Pruned BF
    print("\n2. Pruned Brute Force (with early termination)...")
    start = time.time()
    result_pruned = pruned_brute_force_unified(instance, timeout)
    time_pruned = time.time() - start
    results['pruned'] = (result_pruned, time_pruned)
    stats = {
        'explored': result_pruned.get('explored_solutions', 0),
        'complete': result_pruned.get('complete_solutions_evaluated', 0),
        'pruned': result_pruned.get('branches_pruned', 0)
    }
    print(f"   Time: {time_pruned:.4f}s, Explored: {stats['explored']}, Complete: {stats['complete']}, Pruned: {stats['pruned']}")
    
    # Hybrid DP
    print("\n3. Hybrid (DP for decisions)...")
    start = time.time()
    result_hybrid = hybrid_dp_unified(instance, timeout)
    time_hybrid = time.time() - start
    results['hybrid'] = (result_hybrid, time_hybrid)
    print(f"   Time: {time_hybrid:.4f}s, Routes: {result_hybrid.get('routes_explored', 'N/A')}, DP calls: {result_hybrid.get('dp_executions', 'N/A')}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Algorithm':<20} {'Time (s)':<12} {'Final Capital':<15} {'Speedup':<10}")
    print("-"*80)
    
    for name, (result, time_taken) in results.items():
        capital = result.get('final_capital', 'N/A')
        speedup = time_pure / time_taken if time_taken > 0 else float('inf')
        print(f"{name:<20} {time_taken:<12.4f} {str(capital):<15} {speedup:<10.2f}x")
    
    # Verify all find same solution
    capitals = [r[0].get('final_capital') for r in results.values() if r[0].get('final_capital') is not None]
    if len(set(capitals)) == 1:
        print(f"\n✓ All algorithms found the same optimal solution: {capitals[0]}")
    else:
        print(f"\n⚠ Solutions differ: {capitals}")
    
    return results


if __name__ == "__main__":
    from pure_brute_force import generate_test_instance
    
    # Test with different sizes
    for n in [4, 5, 6, 7, 8, 9, 10]:
        print(f"\n{'='*80}")
        print(f"Testing with {n} ports (including Amsterdam)")
        print(f"{'='*80}")
        
        instance = generate_test_instance(num_ports=n, seed=42)
        compare_all_unified(instance, timeout=60.0)

