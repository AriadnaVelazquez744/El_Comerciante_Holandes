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
    
    Args:
        route: List of port indices [v0, v1, ..., vL, v0]
        instance: Problem instance dictionary
        
    Returns:
        Upper bound on achievable profit
    """
    purchase_prices = instance['purchase_prices']
    sale_prices = instance['sale_prices']
    travel_costs = instance['travel_costs']
    capacity = instance['capacity']
    max_units_per_op = instance.get('max_units_per_op', 1)
    
    total_prize = 0.0
    total_cost = 0.0
    
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
                max_profit_per_port = max(max_profit_per_port, profit_per_unit * max_units)
        
        total_prize += max_profit_per_port
    
    # Calculate total travel cost
    for i in range(len(route) - 1):
        total_cost += travel_costs[route[i]][route[i + 1]]
    
    return total_prize - total_cost


def generate_promising_routes(instance: Dict, beam_width: int = 100, 
                             max_depth: Optional[int] = None, timeout: float = 300.0) -> List[Tuple[List[int], float]]:
    """
    Generate promising routes using beam search with PCTSP-style bounds.
    
    Args:
        instance: Problem instance dictionary
        beam_width: Number of routes to keep in beam at each depth
        max_depth: Maximum route length (None = no limit, but limited by n)
        timeout: Maximum time to spend on route generation
        
    Returns:
        List of (route, bound) tuples, sorted by bound (descending)
    """
    start_time = time.time()
    
    ports = instance['ports']
    n = len(ports)
    travel_costs = instance['travel_costs']
    travel_times = instance['travel_times']
    T_max = instance['max_time']
    
    if max_depth is None:
        max_depth = n - 1  # Can visit at most n-1 ports (excluding Amsterdam)
    
    # Beam: list of (route, bound, time_used, visited_set)
    # Start with just Amsterdam
    beam = [([0], 0.0, 0.0, {0})]
    final_routes = []  # Complete routes ending at Amsterdam
    best_known_bound = float('-inf')
    
    threshold = 0.0  # Pruning threshold (can be adjusted)
    
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
                bound = calculate_pctsp_bound(new_route + [0], instance)
                
                # Pruning: only keep if bound is promising
                if bound > best_known_bound - threshold:
                    candidates.append((new_route, bound, new_time, visited | {next_port}))
            
            # Also consider completing route back to Amsterdam
            if len(route) > 1:  # At least one port visited
                return_time = travel_times[route[-1]][0]
                final_time = time_used + return_time
                
                if final_time <= T_max:
                    complete_route = route + [0]
                    final_bound = calculate_pctsp_bound(complete_route, instance)
                    final_routes.append((complete_route, final_bound))
        
        # Update best known bound
        if candidates:
            best_known_bound = max(best_known_bound, max(c[1] for c in candidates))
        
        # Keep top K candidates by bound
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_width]
        
        if not beam:
            break
    
    # Sort final routes by bound (descending)
    final_routes.sort(key=lambda x: x[1], reverse=True)
    
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
    max_units_per_op = instance.get('max_units_per_op', 1)
    T_max = instance['max_time']
    
    # Determine number of item types from first port
    first_port = route[1] if L > 0 else 0
    num_items = len(purchase_prices[first_port]) if first_port > 0 else 0
    
    if num_items == 0:
        return None, None
    
    # Check time feasibility
    total_time = sum(travel_times[route[i]][route[i+1]] for i in range(len(route) - 1))
    if total_time > T_max:
        return None, None
    
    # State: (port_index, capital, load_vector)
    # Use sparse representation: dp[port_idx][capital_band][load_tuple] = best_capital
    # For efficiency, we discretize capital into bands
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
    initial_load = tuple([0] * num_items)
    initial_band = get_capital_band(capital_after_travel)
    dp[0][(initial_band, initial_load)] = (capital_after_travel, None, None)
    
    # Process each port
    for i in range(L):
        port = route[i + 1]
        dp_next = defaultdict(lambda: (float('-inf'), None, None))
        
        # Process each state at current port
        for (cap_band, load_vec), (best_cap, prev_state, prev_decision) in dp[i].items():
            if best_cap == float('-inf'):
                continue
            
            capital = get_capital_from_band(cap_band)
            
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
                new_capital, new_load = apply_transaction(capital, load_vec, action, port, instance)
                
                if not is_feasible_state(new_capital, new_load, capacity, instance):
                    continue
                
                new_band = get_capital_band(new_capital)
                new_state = (new_band, new_load)
                
                # Update if this is better
                current_best, _, _ = dp_next[new_state]
                if new_capital > current_best:
                    dp_next[new_state] = (new_capital, (cap_band, load_vec), (i, port, action))
        
        # Travel to next port
        if i < L - 1:
            next_port = route[i + 2]
        else:
            next_port = 0  # Return to Amsterdam
        
        travel_cost = travel_costs[port][next_port]
        
        # Update dp[i+1] after travel
        for (cap_band, load_vec), (best_cap, prev_state, prev_decision) in dp_next.items():
            capital = get_capital_from_band(cap_band)
            new_capital = capital - travel_cost
            
            if new_capital < 0:
                continue
            
            new_band = get_capital_band(new_capital)
            new_state = (new_band, load_vec)
            
            current_best, _, _ = dp[i + 1][new_state]
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
            - 'max_units_per_op': Maximum units per operation (default: 1)
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
    
    # Phase 1: Generate promising routes
    route_timeout = timeout * 0.3  # Use 30% of timeout for route generation
    routes = generate_promising_routes(instance, beam_width=beam_width, 
                                      timeout=route_timeout)
    stats['routes_generated'] = len(routes)
    
    # Phase 2: Solve transaction DP for each route
    remaining_time = timeout - (time.time() - start_time)
    
    for route, bound in routes:
        if time.time() - start_time > timeout:
            stats['timeout'] = True
            break
        
        # Early pruning: if bound is worse than current best, skip
        if bound < best_solution['capital']:
            continue
        
        stats['routes_evaluated'] += 1
        max_capital, decisions = solve_transactions_dp(route, instance)
        
        if max_capital is not None and max_capital > best_solution['capital']:
            best_solution['capital'] = max_capital
            best_solution['route'] = route
            best_solution['decisions'] = decisions
    
    execution_time = time.time() - start_time
    
    return {
        'capital': best_solution['capital'] if best_solution['capital'] != float('-inf') else None,
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

