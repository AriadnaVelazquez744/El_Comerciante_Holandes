"""
Two-Phase Hybrid Algorithm for the Dutch Merchant Problem

This module implements an efficient algorithm that separates route generation
(using PCTSP-inspired heuristics) from transaction optimization (using multi-dimensional DP).

Designed to handle instances with:
- 15 <= n <= 30 ports
- m items per port
- k units per buy/sell operation
"""

import math
import time
import heapq
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


def is_feasible_state(capital: float, load_vector: Tuple[int, ...], capacity: int, 
                      instance: Dict) -> bool:
    """
    Check if a state (capital, load_vector) is feasible.
    
    Args:
        capital: Current capital
        load_vector: Tuple of (load_1, load_2, ..., load_m) for m item types
        capacity: Maximum total capacity B
        instance: Problem instance dictionary
        
    Returns:
        True if state is feasible, False otherwise
    """
    if capital < 0:
        return False
    
    total_load = sum(load_vector)
    if total_load > capacity:
        return False
    
    # All loads must be non-negative
    if any(load < 0 for load in load_vector):
        return False
    
    return True


def apply_transaction(capital: float, load_vector: Tuple[int, ...], 
                     action: Tuple[int, int], port: int, instance: Dict) -> Tuple[float, Tuple[int, ...]]:
    """
    Apply a transaction action to a state.
    
    Args:
        capital: Current capital
        load_vector: Current load vector (load_1, ..., load_m)
        action: (item_index, units) where units > 0 means buy, units < 0 means sell
        port: Current port index
        instance: Problem instance dictionary
        
    Returns:
        (new_capital, new_load_vector) after applying action
    """
    item_index, units = action
    purchase_prices = instance['purchase_prices']
    sale_prices = instance['sale_prices']
    
    load_list = list(load_vector)
    new_capital = capital
    new_load = load_list.copy()
    
    if units > 0:  # Buy
        price_per_unit = purchase_prices[port][item_index]
        total_cost = price_per_unit * units
        new_capital = capital - total_cost
        new_load[item_index] = load_list[item_index] + units
    elif units < 0:  # Sell
        units_to_sell = abs(units)
        price_per_unit = sale_prices[port][item_index]
        total_revenue = price_per_unit * units_to_sell
        new_capital = capital + total_revenue
        new_load[item_index] = load_list[item_index] - units_to_sell
    
    return new_capital, tuple(new_load)


def calculate_pctsp_bound(route: List[int], instance: Dict) -> float:
    """
    Calculate PCTSP-style upper bound for a route.
    
    This bound estimates maximum achievable profit by:
    - Summing maximum possible profit per port (prize)
    - Subtracting travel costs
    - Accounting for capital constraints (can't buy if insufficient capital)
    
    Args:
        route: List of port indices [v0, v1, ..., vL, v0]
        instance: Problem instance dictionary
        
    Returns:
        Upper bound on achievable profit (not including initial capital)
    """
    purchase_prices = instance['purchase_prices']
    sale_prices = instance['sale_prices']
    travel_costs = instance['travel_costs']
    capacity = instance['capacity']
    initial_capital = instance['initial_capital']
    max_units_per_op = instance.get('max_units_per_op', capacity)
    
    total_prize = 0.0
    total_cost = 0.0
    current_capital = initial_capital
    
    # Handle empty route [0, 0] - no travel, no profit
    if len(route) <= 2 and route[0] == 0 and route[-1] == 0:
        return 0.0
    
    # Calculate total travel cost first
    for i in range(len(route) - 1):
        total_cost += travel_costs[route[i]][route[i + 1]]
    
    # Simulate capital flow to get more accurate bound
    # Calculate prize (maximum profit) for each port in route
    for i in range(1, len(route) - 1):  # Skip start and end (Amsterdam)
        port = route[i]
        max_profit_per_port = 0.0
        
        # For each item type, calculate maximum profit
        num_items = len(purchase_prices[port])
        for item_idx in range(num_items):
            profit_per_unit = sale_prices[port][item_idx] - purchase_prices[port][item_idx]
            if profit_per_unit > 0:
                # Maximum units we could profitably trade (limited by capacity and k)
                max_units = min(max_units_per_op, capacity)
                # Check if we have enough capital to buy
                purchase_cost = purchase_prices[port][item_idx] * max_units
                if current_capital >= purchase_cost:
                    profit = profit_per_unit * max_units
                    max_profit_per_port = max(max_profit_per_port, profit)
        
        total_prize += max_profit_per_port
        # Update capital (optimistic: assume we make profit)
        current_capital += max_profit_per_port
    
    return total_prize - total_cost


def generate_promising_routes(instance: Dict, beam_width: int = 100, 
                             max_depth: Optional[int] = None, timeout: float = 300.0,
                             adaptive_beam: bool = True) -> List[Tuple[List[int], float]]:
    """
    Generate promising routes using beam search with PCTSP-style bounds.
    
    Args:
        instance: Problem instance dictionary
        beam_width: Number of routes to keep in beam at each depth
        max_depth: Maximum route length (None = no limit, but limited by n)
        timeout: Maximum time to spend on route generation
        adaptive_beam: If True, increase beam width for small instances
        
    Returns:
        List of (route, bound) tuples, sorted by bound (descending)
    """
    start_time = time.time()
    
    ports = instance['ports']
    n = len(ports)
    travel_costs = instance['travel_costs']
    travel_times = instance['travel_times']
    T_max = instance['max_time']
    initial_capital = instance['initial_capital']
    
    # Adaptive beam width: increase for small instances to find better solutions
    if adaptive_beam and n <= 8:
        beam_width = max(beam_width, n * 20)  # More routes for small instances
    
    if max_depth is None:
        max_depth = n - 1  # Can visit at most n-1 ports (excluding Amsterdam)
    
    # Handle edge case: n=1 (only Amsterdam, no other ports)
    if n == 1:
        # Only route is [0, 0] - stay at Amsterdam
        route_00 = [0, 0]
        profit_bound = calculate_pctsp_bound(route_00, instance)
        final_bound = initial_capital + profit_bound
        return [(route_00, final_bound)]
    
    # Beam: list of (route, bound, time_used, visited_set)
    # Start with just Amsterdam
    beam = [([0], initial_capital, 0.0, {0})]  # Bound includes initial capital
    final_routes = []  # Complete routes ending at Amsterdam
    # Initialize best_known_bound to allow all routes initially (very permissive)
    best_known_bound = float('-inf')
    
    # Adaptive threshold: be less aggressive for small instances
    # Positive threshold means we allow routes with bound up to threshold less than best
    if n <= 8:
        threshold = 50.0  # Allow routes with bound up to 50 less than best
    else:
        threshold = 0.0
    
    # Also consider the empty route (just return to Amsterdam immediately)
    # This handles cases where no profitable routes exist
    empty_route = [0, 0]
    empty_profit_bound = calculate_pctsp_bound(empty_route, instance)
    empty_bound = initial_capital + empty_profit_bound
    final_routes.append((empty_route, empty_bound))
    best_known_bound = max(best_known_bound, empty_bound)
    
    for depth in range(1, max_depth + 1):
        if time.time() - start_time > timeout:
            break
            
        candidates = []
        
        for state in beam:
            route = state[0]
            time_used = state[2]
            visited = state[3]
            last_port = route[-1]
            
            # Try extending to each unvisited port
            for next_port in range(1, n):  # Skip Amsterdam (port 0)
                if next_port in visited:
                    continue
                
                travel_time = travel_times[last_port][next_port]
                new_time = time_used + travel_time
                
                if new_time > T_max:
                    continue
                
                new_route = route + [next_port]
                profit_bound = calculate_pctsp_bound(new_route + [0], instance)
                # Bound is profit, convert to capital (initial + profit)
                bound = initial_capital + profit_bound
                
                # Pruning: only keep if bound is promising
                # For small instances, be less aggressive (allow routes with lower bounds)
                if bound >= best_known_bound - threshold:
                    candidates.append((new_route, bound, new_time, visited | {next_port}))
            
            # Also consider completing route back to Amsterdam
            if len(route) > 1:  # At least one port visited
                return_time = travel_times[route[-1]][0]
                final_time = time_used + return_time
                
                if final_time <= T_max:
                    complete_route = route + [0]
                    profit_bound = calculate_pctsp_bound(complete_route, instance)
                    # Convert profit bound to capital bound
                    final_bound = initial_capital + profit_bound
                    final_routes.append((complete_route, final_bound))
        
        # Update best known bound from both candidates and final routes
        if candidates:
            best_known_bound = max(best_known_bound, max(c[1] for c in candidates))
        if final_routes:
            best_known_bound = max(best_known_bound, max(r[1] for r in final_routes))
        
        # Keep top K candidates by bound
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_width]
        
        if not beam:
            break
    
    # Sort final routes by bound (descending)
    final_routes.sort(key=lambda x: x[1], reverse=True)
    
    # Ensure we return at least the empty route if no other routes were found
    if not final_routes:
        empty_route = [0, 0]
        empty_profit_bound = calculate_pctsp_bound(empty_route, instance)
        empty_bound = initial_capital + empty_profit_bound
        final_routes.append((empty_route, empty_bound))
    
    return final_routes


def solve_transactions_dp(route: List[int], instance: Dict) -> Tuple[Optional[float], Optional[List[Dict]]]:
    """
    Solve transaction optimization for a fixed route using multi-dimensional DP.
    
    Args:
        route: Fixed route [v0, v1, ..., vL, v0]
        instance: Problem instance dictionary
        
    Returns:
        (max_capital, decisions_sequence) or (None, None) if infeasible
        decisions_sequence is a list of dicts: [{'port': i, 'item': j, 'action': units}, ...]
    """
    if len(route) < 3:  # Need at least [0, port, 0]
        return None, None
    
    L = len(route) - 2  # Number of intermediate ports
    purchase_prices = instance['purchase_prices']
    sale_prices = instance['sale_prices']
    travel_costs = instance['travel_costs']
    travel_times = instance['travel_times']
    initial_capital = instance['initial_capital']
    capacity = instance['capacity']
    # Default k = capacity if not specified
    max_units_per_op = instance.get('max_units_per_op', capacity)
    T_max = instance['max_time']
    
    # Determine number of item types from first port
    first_port = route[1] if L > 0 else 0
    num_items = len(purchase_prices[first_port]) if first_port > 0 else 0
    
    if num_items == 0:
        return None, None
    
    # Check time feasibility (travel time only - operations will be tracked in DP)
    total_travel_time = sum(travel_times[route[i]][route[i+1]] for i in range(len(route) - 1))
    if total_travel_time > T_max:
        return None, None
    
    # State: (port_index, capital, load_vector, time_used)
    # Use sparse representation: dp[port_idx][(capital_band, load_tuple, time_used)] = best_capital
    # For efficiency, we discretize capital into bands
    # For small instances, use exact capital to avoid precision loss
    n = len(instance['ports'])
    if n <= 8 and num_items == 1:
        capital_band_width = 0.01  # Very fine discretization for small instances
    else:
        capital_band_width = max(1.0, initial_capital / 100.0)  # Adaptive discretization
    
    def get_capital_band(capital: float) -> int:
        """Convert capital to band index"""
        return int(capital / capital_band_width)
    
    def get_capital_from_band(band: int) -> float:
        """Get representative capital value from band"""
        return band * capital_band_width
    
    # Initialize: Start at Amsterdam, travel to first port
    first_port = route[1]
    travel_cost_to_first = travel_costs[0][first_port]
    capital_after_travel = initial_capital - travel_cost_to_first
    
    if capital_after_travel < 0:
        return None, None
    
    # DP table: dp[port_idx] = dict mapping (capital_band, load_tuple) -> (best_capital, prev_state, decision)
    dp = [defaultdict(lambda: (float('-inf'), None, None)) for _ in range(L + 1)]
    
    # Initialize state at first port
    # State includes: (capital_band, load_tuple, time_used)
    initial_load = tuple([0] * num_items)
    initial_band = get_capital_band(capital_after_travel)
    initial_time = travel_times[0][first_port]  # Time to travel to first port
    dp[0][(initial_band, initial_load, initial_time)] = (capital_after_travel, None, None)
    
    # Process each port
    for i in range(L):
        port = route[i + 1]
        dp_next = defaultdict(lambda: (float('-inf'), None, None))
        
        # Process each state at current port
        for (cap_band, load_vec, time_used), (best_cap, prev_state, prev_decision) in dp[i].items():
            if best_cap == float('-inf'):
                continue
            
            capital = get_capital_from_band(cap_band)
            
            # Check if we've already exceeded time limit
            if time_used > T_max:
                continue
            
            # Generate all feasible actions
            # For simplicity, we process items sequentially (can be optimized for independence)
            actions_to_consider = []
            
            # For each item type
            for item_idx in range(num_items):
                current_load = load_vec[item_idx]
                total_current_load = sum(load_vec)
                
                # Option 1: Do nothing
                actions_to_consider.append((item_idx, 0))
                
                # Option 2: Buy operations
                for units in range(1, max_units_per_op + 1):
                    if total_current_load + units > capacity:
                        break
                    price_per_unit = purchase_prices[port][item_idx]
                    if capital >= price_per_unit * units:
                        actions_to_consider.append((item_idx, units))
                
                # Option 3: Sell operations
                for units in range(1, min(max_units_per_op, current_load) + 1):
                    actions_to_consider.append((item_idx, -units))
            
            # Apply each action and update dp_next
            for action in actions_to_consider:
                item_idx, units = action
                
                # Pruning: Check constraints before applying transaction
                if units > 0:  # Buying
                    total_current_load = sum(load_vec)
                    if total_current_load + units > capacity:
                        continue  # Prune: would exceed capacity
                    # Pruning: Check budget constraint
                    price_per_unit = purchase_prices[port][item_idx]
                    total_cost = price_per_unit * units
                    if capital < total_cost:
                        continue  # Prune: insufficient capital
                
                elif units < 0:  # Selling
                    current_load = load_vec[item_idx]
                    units_to_sell = abs(units)
                    if current_load < units_to_sell:
                        continue  # Prune: trying to sell more than available
                
                new_capital, new_load = apply_transaction(capital, load_vec, action, port, instance)
                
                # Additional feasibility check after transaction
                if not is_feasible_state(new_capital, new_load, capacity, instance):
                    continue
                
                # Pruning: Reject if capital becomes negative
                if new_capital < 0:
                    continue
                
                # Track operation time: +1 for each buy/sell operation
                operation_time = 1 if units != 0 else 0
                new_time = time_used + operation_time
                
                # Pruning: Reject if exceeds time limit
                if new_time > T_max:
                    continue
                
                new_band = get_capital_band(new_capital)
                new_state = (new_band, new_load, new_time)
                
                # Update if this is better
                current_best, _, _ = dp_next.get(new_state, (float('-inf'), None, None))
                if new_capital > current_best:
                    dp_next[new_state] = (new_capital, (cap_band, load_vec, time_used), (i, port, action))
        
        # Travel to next port
        if i < L - 1:
            next_port = route[i + 2]
        else:
            next_port = 0  # Return to Amsterdam
        
        travel_cost = travel_costs[port][next_port]
        travel_time_cost = travel_times[port][next_port]
        
        # Update dp[i+1] after travel
        for (cap_band, load_vec, time_after_ops), (best_cap, prev_state, prev_decision) in dp_next.items():
            capital = get_capital_from_band(cap_band)
            new_capital = capital - travel_cost
            
            if new_capital < 0:
                continue
            
            # Add travel time
            new_time = time_after_ops + travel_time_cost
            
            # Pruning: Reject if exceeds time limit
            if new_time > T_max:
                continue
            
            new_band = get_capital_band(new_capital)
            new_state = (new_band, load_vec, new_time)
            
            current_best, _, _ = dp[i + 1].get(new_state, (float('-inf'), None, None))
            if new_capital > current_best:
                dp[i + 1][new_state] = (new_capital, prev_state, prev_decision)
    
    # Find best final state
    best_final_capital = float('-inf')
    best_final_state = None
    
    for state, (capital, prev_state, prev_decision) in dp[L].items():
        if capital > best_final_capital:
            best_final_capital = capital
            best_final_state = (state, prev_state, prev_decision)
    
    if best_final_capital == float('-inf'):
        return None, None
    
    # Reject solutions with final capital less than initial capital (not valid)
    if best_final_capital < initial_capital:
        return None, None
    
    # Reconstruct decisions by backtracking
    decisions = []
    current_state_key = best_final_state[0]
    
    # Backtrack through ports
    for i in range(L - 1, -1, -1):
        if current_state_key is None:
            break
        
        state_data = dp[i + 1].get(current_state_key)
        if state_data is None:
            break
        
        best_cap, prev_state, decision = state_data
        
        if decision is not None:
            port_idx, port, action = decision
            item_idx, units = action
            decisions.insert(0, {
                'port': port,
                'item': item_idx,
                'action': units
            })
        
        # Move to previous state
        if prev_state is not None:
            current_state_key = prev_state
        else:
            break
    
    return best_final_capital, decisions


def two_phase_hybrid_solve(instance: Dict, timeout: float = 300.0, 
                           beam_width: int = 100) -> Dict:
    """
    Main two-phase hybrid algorithm.
    
    Args:
        instance: Problem instance dictionary with extended format:
            - 'ports': List of port names
            - 'travel_costs': 2D list of travel costs
            - 'travel_times': 2D list of travel times
            - 'purchase_prices': Dict[port_index, List[price_per_item]]
            - 'sale_prices': Dict[port_index, List[price_per_item]]
            - 'initial_capital': Initial capital
            - 'capacity': Maximum capacity
            - 'max_time': Maximum time
            - 'num_items': Number of item types (optional, inferred)
            - 'max_units_per_op': Maximum units per operation (default: capacity)
        timeout: Maximum execution time in seconds
        beam_width: Beam width for route generation
        
    Returns:
        Dictionary with:
            - 'capital': Best capital found (or -inf if no solution)
            - 'route': Best route found
            - 'decisions': Decision sequence
            - 'routes_generated': Number of routes generated
            - 'routes_evaluated': Number of routes evaluated with DP
            - 'execution_time': Total execution time
            - 'timeout': Whether timeout was reached
    """
    start_time = time.time()
    
    # Statistics
    stats = {
        'routes_generated': 0,
        'routes_evaluated': 0,
        'timeout': False
    }
    
    # Best solution
    best_solution = {
        'capital': float('-inf'),
        'route': None,
        'decisions': None
    }
    
    # Adaptive parameters based on instance size
    n = len(instance['ports'])
    initial_capital = instance['initial_capital']
    
    # For small instances, evaluate more routes
    if n <= 8:
        # Increase beam width significantly for small instances
        adaptive_beam_width = max(beam_width, n * 30)
        # Less aggressive pruning - evaluate routes even if bound is slightly lower
        bound_tolerance = 20.0
        max_routes_to_evaluate = min(100, 2 ** (n - 1))  # More routes for small n
    else:
        adaptive_beam_width = beam_width
        bound_tolerance = 0.0
        max_routes_to_evaluate = 200
    
    # Iterative route generation with DP feedback
    # Round 1: Initial route generation
    route_timeout = timeout * 0.3  # Use 30% of timeout for initial route generation
    routes = generate_promising_routes(instance, beam_width=adaptive_beam_width, 
                                      timeout=route_timeout, adaptive_beam=True)
    stats['routes_generated'] = len(routes)
    
    # Track evaluated routes and their DP results for refinement
    evaluated_routes = {}  # route_tuple -> (capital, decisions)
    route_scores = []  # List of (route, capital) for sorting
    
    # Phase 2: Solve transaction DP for each route (initial round)
    routes_evaluated_count = 0
    for route, bound in routes:
        if time.time() - start_time > timeout:
            stats['timeout'] = True
            break
        
        # Adaptive pruning: for small instances, be less aggressive
        # Compare bound (which now includes initial capital) to current best
        # Only prune if we have a valid solution and bound is significantly worse
        if best_solution['capital'] != float('-inf'):
            if bound < best_solution['capital'] - bound_tolerance:
                continue
        
        # Limit number of routes evaluated for very large instances
        if routes_evaluated_count >= max_routes_to_evaluate:
            break
        
        routes_evaluated_count += 1
        stats['routes_evaluated'] += 1
        max_capital, decisions = solve_transactions_dp(route, instance)
        
        route_tuple = tuple(route)
        evaluated_routes[route_tuple] = (max_capital, decisions)
        
        if max_capital is not None:
            route_scores.append((route, max_capital))
            if max_capital > best_solution['capital']:
                best_solution['capital'] = max_capital
                best_solution['route'] = route
                best_solution['decisions'] = decisions
    
    # Round 2: Generate additional routes based on successful patterns
    # If we have good solutions, try variations of successful routes
    if route_scores and time.time() - start_time < timeout * 0.8:
        # Sort routes by capital (descending)
        route_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top routes and generate neighbors (add/remove one port)
        top_routes = route_scores[:min(5, len(route_scores))]
        additional_routes = []
        
        for route, capital in top_routes:
            if time.time() - start_time > timeout * 0.8:
                break
            
            # Try adding one unvisited port to successful routes
            visited_ports = set(route[1:-1])  # Exclude start and end (Amsterdam)
            n = len(instance['ports'])
            
            for new_port in range(1, n):
                if new_port in visited_ports:
                    continue
                
                # Try inserting at different positions
                for insert_pos in range(1, len(route)):
                    new_route = route[:insert_pos] + [new_port] + route[insert_pos:-1] + [0]
                    route_tuple = tuple(new_route)
                    
                    # Skip if already evaluated
                    if route_tuple in evaluated_routes:
                        continue
                    
                    # Check time feasibility quickly
                    total_time = sum(instance['travel_times'][new_route[i]][new_route[i+1]] 
                                   for i in range(len(new_route) - 1))
                    if total_time <= instance['max_time']:
                        profit_bound = calculate_pctsp_bound(new_route, instance)
                        bound = initial_capital + profit_bound
                        # Use adaptive tolerance
                        if bound > best_solution['capital'] - bound_tolerance:
                            additional_routes.append((new_route, bound))
            
            # Try removing one port from successful routes (if route has more than 1 port)
            if len(route) > 3:  # More than just [0, port, 0]
                for remove_pos in range(1, len(route) - 1):
                    new_route = route[:remove_pos] + route[remove_pos+1:]
                    route_tuple = tuple(new_route)
                    
                    if route_tuple in evaluated_routes:
                        continue
                    
                    total_time = sum(instance['travel_times'][new_route[i]][new_route[i+1]] 
                                   for i in range(len(new_route) - 1))
                    if total_time <= instance['max_time']:
                        profit_bound = calculate_pctsp_bound(new_route, instance)
                        bound = initial_capital + profit_bound
                        if bound > best_solution['capital'] - bound_tolerance:
                            additional_routes.append((new_route, bound))
        
        # Sort additional routes by bound
        additional_routes.sort(key=lambda x: x[1], reverse=True)
        stats['routes_generated'] += len(additional_routes)
        
        # Evaluate additional routes
        for route, bound in additional_routes:
            if time.time() - start_time > timeout:
                stats['timeout'] = True
                break
            
            if bound < best_solution['capital']:
                continue
            
            route_tuple = tuple(route)
            if route_tuple in evaluated_routes:
                continue
            
            stats['routes_evaluated'] += 1
            max_capital, decisions = solve_transactions_dp(route, instance)
            
            evaluated_routes[route_tuple] = (max_capital, decisions)
            
            if max_capital is not None:
                if max_capital > best_solution['capital']:
                    best_solution['capital'] = max_capital
                    best_solution['route'] = route
                    best_solution['decisions'] = decisions
    
    # Round 3: For small instances, generate more routes using DP-informed heuristics
    # Use actual DP results to identify promising port patterns
    if n <= 8 and route_scores and time.time() - start_time < timeout * 0.9:
        # Identify most profitable ports from successful routes
        port_profits = {}  # port -> list of profits achieved
        for route, capital in route_scores[:min(10, len(route_scores))]:
            for port in route[1:-1]:  # Exclude Amsterdam
                if port not in port_profits:
                    port_profits[port] = []
                port_profits[port].append(capital)
        
        # Calculate average profit per port
        port_avg_profit = {p: sum(profits) / len(profits) 
                          for p, profits in port_profits.items()}
        
        # Generate routes prioritizing high-profit ports
        if port_avg_profit:
            sorted_ports = sorted(port_avg_profit.items(), key=lambda x: x[1], reverse=True)
            top_ports = [p for p, _ in sorted_ports[:min(n-1, 5)]]
            
            # Generate routes visiting top profitable ports
            import itertools
            for k in range(1, min(len(top_ports) + 1, 4)):  # Up to 3 ports
                if time.time() - start_time > timeout * 0.9:
                    break
                for subset in itertools.combinations(top_ports, k):
                    for perm in itertools.permutations(subset):
                        new_route = [0] + list(perm) + [0]
                        route_tuple = tuple(new_route)
                        
                        if route_tuple in evaluated_routes:
                            continue
                        
                        # Check time feasibility
                        total_time = sum(instance['travel_times'][new_route[i]][new_route[i+1]] 
                                       for i in range(len(new_route) - 1))
                        if total_time <= instance['max_time']:
                            profit_bound = calculate_pctsp_bound(new_route, instance)
                            bound = initial_capital + profit_bound
                            if bound > best_solution['capital'] - bound_tolerance:
                                if routes_evaluated_count >= max_routes_to_evaluate:
                                    break
                                
                                routes_evaluated_count += 1
                                stats['routes_evaluated'] += 1
                                max_capital, decisions = solve_transactions_dp(new_route, instance)
                                
                                evaluated_routes[route_tuple] = (max_capital, decisions)
                                
                                if max_capital is not None:
                                    if max_capital > best_solution['capital']:
                                        best_solution['capital'] = max_capital
                                        best_solution['route'] = new_route
                                        best_solution['decisions'] = decisions
    
    # Safety check: If no valid solution found (all routes resulted in capital < initial),
    # generate more routes using a different strategy
    final_capital = best_solution['capital'] if best_solution['capital'] != float('-inf') else None
    no_valid_solution = (final_capital is None or final_capital < instance['initial_capital'])
    
    if no_valid_solution and time.time() - start_time < timeout * 0.95:
        # Generate additional routes using exhaustive or relaxed strategy
        # This safety mechanism ensures we explore more possibilities when
        # all previously evaluated routes resulted in invalid solutions
        
        if n <= 8:
            # For small instances, generate routes exhaustively
            import itertools
            emergency_routes = []
            
            # Generate all possible route combinations
            ports_to_visit = list(range(1, n))  # All ports except Amsterdam
            
            for route_length in range(1, min(n, 5) + 1):  # Up to 5 ports
                if time.time() - start_time > timeout * 0.95:
                    break
                    
                for subset in itertools.combinations(ports_to_visit, route_length):
                    for perm in itertools.permutations(subset):
                        new_route = [0] + list(perm) + [0]
                        route_tuple = tuple(new_route)
                        
                        # Skip if already evaluated
                        if route_tuple in evaluated_routes:
                            continue
                        
                        # Check time feasibility
                        total_time = sum(instance['travel_times'][new_route[i]][new_route[i+1]] 
                                       for i in range(len(new_route) - 1))
                        if total_time <= instance['max_time']:
                            profit_bound = calculate_pctsp_bound(new_route, instance)
                            bound = initial_capital + profit_bound
                            emergency_routes.append((new_route, bound))
            
            # Sort by bound and evaluate
            emergency_routes.sort(key=lambda x: x[1], reverse=True)
            stats['routes_generated'] += len(emergency_routes)
            
            for route, bound in emergency_routes[:min(200, len(emergency_routes))]:
                if time.time() - start_time > timeout * 0.95:
                    break
                
                route_tuple = tuple(route)
                if route_tuple in evaluated_routes:
                    continue
                
                stats['routes_evaluated'] += 1
                max_capital, decisions = solve_transactions_dp(route, instance)
                
                evaluated_routes[route_tuple] = (max_capital, decisions)
                
                if max_capital is not None and max_capital >= instance['initial_capital']:
                    if max_capital > best_solution['capital']:
                        best_solution['capital'] = max_capital
                        best_solution['route'] = route
                        best_solution['decisions'] = decisions
                        final_capital = max_capital
                        no_valid_solution = False
                        # Continue searching for better solutions, but we found at least one valid
        
        else:
            # For larger instances, use relaxed beam search with much larger beam width
            relaxed_beam_width = beam_width * 5  # Much larger beam
            relaxed_timeout = min(timeout * 0.1, timeout - (time.time() - start_time))
            
            if relaxed_timeout > 1.0:  # Only if we have enough time
                relaxed_routes = generate_promising_routes(
                    instance, 
                    beam_width=relaxed_beam_width, 
                    timeout=relaxed_timeout,
                    adaptive_beam=False  # Don't further increase
                )
                
                stats['routes_generated'] += len(relaxed_routes)
                
                # Evaluate relaxed routes with no pruning
                for route, bound in relaxed_routes:
                    if time.time() - start_time > timeout * 0.95:
                        break
                    
                    route_tuple = tuple(route)
                    if route_tuple in evaluated_routes:
                        continue
                    
                    stats['routes_evaluated'] += 1
                    max_capital, decisions = solve_transactions_dp(route, instance)
                    
                    evaluated_routes[route_tuple] = (max_capital, decisions)
                    
                    if max_capital is not None and max_capital >= instance['initial_capital']:
                        if max_capital > best_solution['capital']:
                            best_solution['capital'] = max_capital
                            best_solution['route'] = route
                            best_solution['decisions'] = decisions
                            final_capital = max_capital
                            no_valid_solution = False
                            # Continue searching for better solutions
    
    execution_time = time.time() - start_time
    
    # Final validation: reject if capital is less than initial capital
    final_capital = best_solution['capital'] if best_solution['capital'] != float('-inf') else None
    if final_capital is not None and final_capital < instance['initial_capital']:
        final_capital = None
        best_solution['route'] = None
        best_solution['decisions'] = None
    
    return {
        'capital': final_capital,
        'route': best_solution['route'],
        'decisions': best_solution['decisions'],
        'routes_generated': stats['routes_generated'],
        'routes_evaluated': stats['routes_evaluated'],
        'execution_time': execution_time,
        'timeout': stats['timeout']
    }


def generate_multi_item_instance(n_ports: int, m_items: int, k_max_units: int, 
                                 seed: int = 42) -> Dict:
    """
    Generate a test instance with multiple items per port.
    
    Args:
        n_ports: Number of ports (including Amsterdam)
        m_items: Number of item types per port
        k_max_units: Maximum units per buy/sell operation
        seed: Random seed for reproducibility
        
    Returns:
        Instance dictionary in extended format
    """
    import random
    random.seed(seed)
    
    ports = ['Amsterdam'] + [f'Port_{i}' for i in range(1, n_ports)]
    
    # Initialize matrices
    costs = [[0] * n_ports for _ in range(n_ports)]
    travel_times = [[0] * n_ports for _ in range(n_ports)]
    
    # Generate cost and time matrices
    for i in range(n_ports):
        for j in range(i + 1, n_ports):
            cost = random.randint(2, 15)
            time_val = random.randint(1, 8)
            costs[i][j] = cost
            costs[j][i] = cost
            travel_times[i][j] = time_val
            travel_times[j][i] = time_val
    
    # Generate prices for multiple items
    purchase_prices = {}
    sale_prices = {}
    
    for port in range(n_ports):
        if port == 0:  # Amsterdam
            purchase_prices[port] = [0] * m_items
            sale_prices[port] = [0] * m_items
        else:
            purchase_prices[port] = [random.randint(8, 25) for _ in range(m_items)]
            sale_prices[port] = [p + random.randint(10, 30) for p in purchase_prices[port]]
    
    # Parameters
    initial_capital = 100 + (n_ports - 3) * 20
    capacity = min(5, n_ports - 1)
    max_time = 20 + (n_ports - 2) * 10
    
    # Ensure at least one profitable route
    if n_ports > 1:
        # Make port 1 highly profitable
        costs[0][1] = random.randint(3, 8)
        costs[1][0] = costs[0][1]
        travel_times[0][1] = random.randint(1, 4)
        travel_times[1][0] = travel_times[0][1]
        
        # Ensure affordable prices and good margins
        for item_idx in range(m_items):
            if purchase_prices[1][item_idx] > initial_capital - costs[0][1] - costs[1][0] - 20:
                purchase_prices[1][item_idx] = max(8, (initial_capital - costs[0][1] - costs[1][0] - 30) // 2)
            min_margin = 15 + costs[0][1] + costs[1][0]
            if sale_prices[1][item_idx] - purchase_prices[1][item_idx] < min_margin:
                sale_prices[1][item_idx] = purchase_prices[1][item_idx] + min_margin + random.randint(5, 15)
    
    return {
        'ports': ports,
        'travel_costs': costs,
        'travel_times': travel_times,
        'purchase_prices': purchase_prices,
        'sale_prices': sale_prices,
        'initial_capital': initial_capital,
        'capacity': capacity,
        'max_time': max_time,
        'num_items': m_items,
        'max_units_per_op': k_max_units
    }


if __name__ == '__main__':
    # Test with small instance
    print("Testing two-phase hybrid algorithm...")
    instance = generate_multi_item_instance(n_ports=5, m_items=2, k_max_units=2, seed=42)
    result = two_phase_hybrid_solve(instance, timeout=60.0, beam_width=50)
    print(f"Result: {result}")

