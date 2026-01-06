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


def generate_promising_routes(instance: Dict, beam_width: int = 200, 
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


def solve_transactions_dp_total_load(route: List[int], instance: Dict) -> Tuple[Optional[float], Optional[List[Dict]]]:
    """
    Solve transaction optimization using total load discretization (for m >= 8).
    
    This is a CRITICAL OPTIMIZATION that reduces state space from O(B^m) to O(B)
    by tracking only total load instead of per-item loads.
    
    Args:
        route: Fixed route [v0, v1, ..., vL, v0]
        instance: Problem instance dictionary
        
    Returns:
        (max_capital, decisions_sequence) or (None, None) if infeasible
    """
    if len(route) < 3:
        return None, None
    
    L = len(route) - 2
    purchase_prices = instance['purchase_prices']
    sale_prices = instance['sale_prices']
    travel_costs = instance['travel_costs']
    travel_times = instance['travel_times']
    initial_capital = instance['initial_capital']
    capacity = instance['capacity']
    max_units_per_op = instance.get('max_units_per_op', capacity)
    T_max = instance['max_time']
    
    first_port = route[1] if L > 0 else 0
    num_items = len(purchase_prices[first_port]) if first_port > 0 else 0
    
    if num_items == 0:
        return None, None
    
    # Check time feasibility
    total_travel_time = sum(travel_times[route[i]][route[i+1]] for i in range(len(route) - 1))
    if total_travel_time > T_max:
        return None, None
    
    # Adaptive capital discretization for large m
    capital_band_width = max(2.0, initial_capital / 50.0)
    
    def get_capital_band(capital: float) -> int:
        return int(capital / capital_band_width)
    
    def get_capital_from_band(band: int) -> float:
        return band * capital_band_width
    
    # Initialize
    first_port = route[1]
    travel_cost_to_first = travel_costs[0][first_port]
    capital_after_travel = initial_capital - travel_cost_to_first
    
    if capital_after_travel < 0:
        return None, None
    
    # State: (capital_band, total_load, time_used) - MUCH smaller than (capital_band, load_vector, time_used)
    dp = [defaultdict(lambda: (float('-inf'), None, None, None)) for _ in range(L + 1)]
    
    initial_band = get_capital_band(capital_after_travel)
    initial_time = travel_times[0][first_port]
    # Store: (best_capital, prev_state, decision, item_distribution_estimate)
    dp[0][(initial_band, 0, initial_time)] = (capital_after_travel, None, None, tuple([0] * num_items))
    
    # Process each port
    for i in range(L):
        port = route[i + 1]
        dp_next = defaultdict(lambda: (float('-inf'), None, None, None))
        
        for (cap_band, total_load, time_used), (best_cap, prev_state, prev_decision, load_dist) in dp[i].items():
            if best_cap == float('-inf'):
                continue
            
            capital = get_capital_from_band(cap_band)
            
            if time_used > T_max:
                continue
            
            # Generate actions: for each item, consider do nothing, buy 1, sell 1
            # This is a greedy approximation - we don't consider all combinations
            actions_to_consider = []
            
            for item_idx in range(num_items):
                # Estimate current load for this item (from load_dist if available)
                current_item_load = load_dist[item_idx] if load_dist else 0
                
                # Option 1: Do nothing
                actions_to_consider.append((item_idx, 0))
                
                # Option 2: Buy 1 unit (if capacity allows and capital allows)
                if total_load < capacity:
                    price_per_unit = purchase_prices[port][item_idx]
                    if capital >= price_per_unit:
                        actions_to_consider.append((item_idx, 1))
                
                # Option 3: Sell 1 unit (if we have this item)
                if current_item_load > 0:
                    actions_to_consider.append((item_idx, -1))
            
            # Apply each action
            for action in actions_to_consider:
                item_idx, units = action
                
                if units > 0:  # Buying
                    if total_load + units > capacity:
                        continue
                    price_per_unit = purchase_prices[port][item_idx]
                    total_cost = price_per_unit * units
                    if capital < total_cost:
                        continue
                    new_capital = capital - total_cost
                    new_total_load = total_load + units
                    # Update load distribution estimate
                    new_load_dist = list(load_dist) if load_dist else [0] * num_items
                    new_load_dist[item_idx] += units
                    
                elif units < 0:  # Selling
                    if current_item_load < abs(units):
                        continue
                    price_per_unit = sale_prices[port][item_idx]
                    total_revenue = price_per_unit * abs(units)
                    new_capital = capital + total_revenue
                    new_total_load = total_load - abs(units)
                    new_load_dist = list(load_dist) if load_dist else [0] * num_items
                    new_load_dist[item_idx] -= abs(units)
                else:  # Do nothing
                    new_capital = capital
                    new_total_load = total_load
                    new_load_dist = load_dist
                
                if new_capital < 0 or new_total_load < 0 or new_total_load > capacity:
                    continue
                
                operation_time = 1 if units != 0 else 0
                new_time = time_used + operation_time
                
                if new_time > T_max:
                    continue
                
                new_band = get_capital_band(new_capital)
                new_state = (new_band, new_total_load, new_time)
                
                current_best, _, _, _ = dp_next.get(new_state, (float('-inf'), None, None, None))
                if new_capital > current_best:
                    dp_next[new_state] = (new_capital, (cap_band, total_load, time_used), (i, port, action), tuple(new_load_dist))
        
        # Travel to next port
        if i < L - 1:
            next_port = route[i + 2]
        else:
            next_port = 0
        
        travel_cost = travel_costs[port][next_port]
        travel_time_cost = travel_times[port][next_port]
        
        for (cap_band, total_load, time_after_ops), (best_cap, prev_state, prev_decision, load_dist) in dp_next.items():
            capital = get_capital_from_band(cap_band)
            new_capital = capital - travel_cost
            
            if new_capital < 0:
                continue
            
            new_time = time_after_ops + travel_time_cost
            
            if new_time > T_max:
                continue
            
            new_band = get_capital_band(new_capital)
            new_state = (new_band, total_load, new_time)
            
            current_best, _, _, _ = dp[i + 1].get(new_state, (float('-inf'), None, None, None))
            if new_capital > current_best:
                dp[i + 1][new_state] = (new_capital, prev_state, prev_decision, load_dist)
    
    # Find best final state
    best_final_capital = float('-inf')
    best_final_state = None
    
    for state, (capital, prev_state, prev_decision, _) in dp[L].items():
        if capital > best_final_capital:
            best_final_capital = capital
            best_final_state = (state, prev_state, prev_decision)
    
    if best_final_capital == float('-inf') or best_final_capital < initial_capital:
        return None, None
    
    # Reconstruct decisions (simplified - may not be exact due to approximation)
    decisions = []
    current_state_key = best_final_state[0]
    
    for i in range(L - 1, -1, -1):
        if current_state_key is None:
            break
        
        state_data = dp[i + 1].get(current_state_key)
        if state_data is None:
            break
        
        best_cap, prev_state, decision, _ = state_data
        
        if decision is not None:
            port_idx, port, action = decision
            item_idx, units = action
            decisions.insert(0, {
                'port': port,
                'item': item_idx,
                'action': units
            })
        
        if prev_state is not None:
            current_state_key = prev_state
        else:
            break
    
    return best_final_capital, decisions


def solve_transactions_dp_independent(route: List[int], instance: Dict) -> Tuple[Optional[float], Optional[List[Dict]]]:
    """
    Solve transaction optimization using item-independent approximation (for m >= 6).
    
    This is a CRITICAL OPTIMIZATION that solves each item independently and combines results.
    Complexity: O(m * B * L) instead of O(B^m * L)
    
    Args:
        route: Fixed route [v0, v1, ..., vL, v0]
        instance: Problem instance dictionary
        
    Returns:
        (max_capital, decisions_sequence) or (None, None) if infeasible
    """
    if len(route) < 3:
        return None, None
    
    L = len(route) - 2
    purchase_prices = instance['purchase_prices']
    sale_prices = instance['sale_prices']
    travel_costs = instance['travel_costs']
    travel_times = instance['travel_times']
    initial_capital = instance['initial_capital']
    capacity = instance['capacity']
    max_units_per_op = instance.get('max_units_per_op', capacity)
    T_max = instance['max_time']
    
    first_port = route[1] if L > 0 else 0
    num_items = len(purchase_prices[first_port]) if first_port > 0 else 0
    
    if num_items == 0:
        return None, None
    
    # Check time feasibility
    total_travel_time = sum(travel_times[route[i]][route[i+1]] for i in range(len(route) - 1))
    if total_travel_time > T_max:
        return None, None
    
    # Solve each item independently with capacity constraint
    # We allocate capacity proportionally to items based on profit potential
    item_solutions = []  # List of (item_idx, decisions, capital_gain, capacity_used)
    
    # Calculate profit potential for each item
    item_profits = []
    for item_idx in range(num_items):
        total_profit = 0.0
        for i in range(1, len(route) - 1):
            port = route[i]
            profit_per_unit = sale_prices[port][item_idx] - purchase_prices[port][item_idx]
            if profit_per_unit > 0:
                total_profit += profit_per_unit
        item_profits.append((item_idx, total_profit))
    
    # Sort by profit potential (descending)
    item_profits.sort(key=lambda x: x[1], reverse=True)
    
    # Allocate capacity to items (greedy allocation)
    remaining_capacity = capacity
    allocated_capacity = {}  # item_idx -> allocated capacity
    
    for item_idx, profit in item_profits:
        # Allocate capacity proportionally, but at least 1 unit per profitable item
        if profit > 0:
            alloc = max(1, min(remaining_capacity // (num_items - len(allocated_capacity)), capacity // num_items))
            allocated_capacity[item_idx] = alloc
            remaining_capacity -= alloc
    
    # Solve each item independently with its allocated capacity
    total_capital_gain = 0.0
    all_decisions = []
    
    for item_idx, profit in item_profits:
        if item_idx not in allocated_capacity:
            continue
        
        item_capacity = allocated_capacity[item_idx]
        
        # Create a single-item instance for this item
        single_item_instance = {
            'purchase_prices': {port: [purchase_prices[port][item_idx]] for port in purchase_prices},
            'sale_prices': {port: [sale_prices[port][item_idx]] for port in sale_prices},
            'travel_costs': instance['travel_costs'],
            'travel_times': instance['travel_times'],
            'initial_capital': initial_capital,  # Share capital across items
            'capacity': item_capacity,
            'max_time': T_max,
            'max_units_per_op': min(max_units_per_op, item_capacity),
            'ports': instance['ports']
        }
        
        # Solve with single-item DP (recursive call, but with m=1, so it uses full DP)
        # We need to call the full DP with a modified instance
        # For efficiency, we'll use a simplified greedy approach per item
        item_capital_gain, item_decisions = solve_item_greedy(route, item_idx, single_item_instance, initial_capital)
        
        if item_decisions:
            total_capital_gain += item_capital_gain
            # Adjust item index in decisions
            for dec in item_decisions:
                dec['item'] = item_idx
            all_decisions.extend(item_decisions)
            initial_capital += item_capital_gain  # Update capital for next items
    
    # Calculate final capital
    first_port = route[1]
    travel_cost_to_first = travel_costs[0][first_port]
    capital_after_travel = instance['initial_capital'] - travel_cost_to_first
    
    if capital_after_travel < 0:
        return None, None
    
    # Apply travel costs for the route
    total_travel_cost = sum(travel_costs[route[i]][route[i+1]] for i in range(len(route) - 1))
    final_capital = instance['initial_capital'] - total_travel_cost + total_capital_gain
    
    if final_capital < instance['initial_capital']:
        return None, None
    
    # Sort decisions by port order
    all_decisions.sort(key=lambda x: route.index(x['port']) if x['port'] in route else len(route))
    
    return final_capital, all_decisions


def solve_item_greedy(route: List[int], item_idx: int, instance: Dict, available_capital: float) -> Tuple[float, List[Dict]]:
    """
    Greedy solution for a single item along a route.
    This is used by the item-independent approximation.
    """
    L = len(route) - 2
    if L == 0:
        return 0.0, []
    
    purchase_prices = instance['purchase_prices']
    sale_prices = instance['sale_prices']
    capacity = instance['capacity']
    max_units_per_op = instance.get('max_units_per_op', capacity)
    
    decisions = []
    current_load = 0
    current_capital = available_capital
    total_profit = 0.0
    
    # Greedy strategy: buy at first profitable port, sell at best opportunity
    for i in range(1, len(route) - 1):
        port = route[i]
        purchase_price = purchase_prices[port][0]  # Single item
        sale_price = sale_prices[port][0]
        profit_per_unit = sale_price - purchase_price
        
        # If profitable and we have capacity and capital, buy
        if profit_per_unit > 0 and current_load < capacity and current_capital >= purchase_price:
            # Buy as much as possible
            max_buy = min(max_units_per_op, capacity - current_load, int(current_capital / purchase_price))
            if max_buy > 0:
                cost = purchase_price * max_buy
                current_capital -= cost
                current_load += max_buy
                decisions.append({
                    'port': port,
                    'item': item_idx,
                    'action': max_buy
                })
        
        # If we have items, consider selling
        if current_load > 0:
            # Sell all if profitable
            if profit_per_unit > 0:
                sell_amount = min(max_units_per_op, current_load)
                revenue = sale_price * sell_amount
                current_capital += revenue
                current_load -= sell_amount
                total_profit += profit_per_unit * sell_amount
                decisions.append({
                    'port': port,
                    'item': item_idx,
                    'action': -sell_amount
                })
    
    return total_profit, decisions


def solve_transactions_dp(route: List[int], instance: Dict) -> Tuple[Optional[float], Optional[List[Dict]]]:
    """
    Solve transaction optimization for a fixed route using multi-dimensional DP.
    
    Uses different strategies based on number of items (CRITICAL OPTIMIZATIONS):
    - m >= 8: Total load discretization (O(B) states, fastest, approximate)
    - 6 <= m < 8: Item-independent approximation (O(m*B) states, faster, approximate)
    - m < 6: Full multi-dimensional DP (O(B^m) states, exact, slower)
    
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
    
    # CRITICAL IMPROVEMENT: Use total load discretization for very large m (check first)
    if num_items >= 8:
        return solve_transactions_dp_total_load(route, instance)
    
    # CRITICAL IMPROVEMENT: Use item-independent approximation for large m
    if num_items >= 6:
        return solve_transactions_dp_independent(route, instance)
    
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
        
        # CRITICAL IMPROVEMENT: Dominance pruning before travel
        # Remove dominated states: if state A has more capital and same/less load than state B, remove B
        if len(dp_next) > 1000:  # Only apply dominance pruning if state space is large
            pruned_dp_next = {}
            states_list = list(dp_next.items())
            states_list.sort(key=lambda x: (sum(x[0][1]), -x[1][0]))  # Sort by total load, then capital (descending)
            
            for state, value in states_list:
                cap_band, load_vec, time_used = state
                best_cap, prev_state, prev_decision = value
                total_load = sum(load_vec)
                
                # Check if this state is dominated
                is_dominated = False
                for existing_state, existing_value in pruned_dp_next.items():
                    existing_cap_band, existing_load_vec, existing_time = existing_state
                    existing_cap, _, _ = existing_value
                    existing_total_load = sum(existing_load_vec)
                    
                    # State is dominated if: same time, less capital, same or more load
                    if (existing_time == time_used and 
                        existing_cap >= best_cap and 
                        existing_total_load <= total_load and
                        all(existing_load_vec[j] <= load_vec[j] for j in range(len(load_vec)))):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    # Remove states that are dominated by this one
                    pruned_dp_next = {s: v for s, v in pruned_dp_next.items() 
                                     if not (s[2] == time_used and 
                                            best_cap >= v[0] and 
                                            total_load <= sum(s[1]) and
                                            all(load_vec[j] <= s[1][j] for j in range(len(load_vec))))}
                    pruned_dp_next[state] = value
            
            dp_next = pruned_dp_next
        
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


def two_phase_hybrid_solve(instance: Dict, timeout: float = 500.0, 
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
    
    # CRITICAL IMPROVEMENT: Adaptive parameters based on instance size
    # For large instances (n >= 15), eliminate route cycling and reduce route evaluation
    if n <= 8:
        # Increase beam width significantly for small instances
        adaptive_beam_width = max(beam_width, n * 30)
        # Less aggressive pruning - evaluate routes even if bound is slightly lower
        bound_tolerance = 20.0
        max_routes_to_evaluate = min(100, 2 ** (n - 1))  # More routes for small n
    elif n >= 15:
        # For large instances: single-pass only, no cycling, fewer routes
        adaptive_beam_width = beam_width
        bound_tolerance = 0.0
        max_routes_to_evaluate = min(50, beam_width * 2)  # Much fewer routes for large n
    else:
        # Medium instances: moderate evaluation
        adaptive_beam_width = beam_width
        bound_tolerance = 0.0
        max_routes_to_evaluate = 100
    
    # Iterative route generation with DP feedback
    # Round 1: Initial route generation
    # CRITICAL IMPROVEMENT: For large n, use more time for initial route generation (single-pass)
    if n >= 15:
        route_timeout = timeout * 0.5  # Use 50% of timeout for large instances (single-pass only)
    else:
        route_timeout = timeout * 0.3  # Use 30% of timeout for initial route generation
    routes = generate_promising_routes(instance, beam_width=adaptive_beam_width, 
                                      timeout=route_timeout, adaptive_beam=True)
    stats['routes_generated'] = len(routes)
    
    # Track evaluated routes and their DP results for refinement
    evaluated_routes = {}  # route_tuple -> (capital, decisions)
    route_scores = []  # List of (route, capital) for sorting
    
    # CONTINUOUS CYCLE: Process routes in batches of 50 until valid solution found
    # This ensures we systematically evaluate routes while reusing memorized (cached) results
    # Continues up to 6 cycles (300 routes max) or until valid solution found or routes exhausted
    
    # Separate trivial and non-trivial routes, prioritize non-trivial
    non_trivial_routes = []
    trivial_routes = []
    
    for route, bound in routes:
        if len(route) <= 2 and route[0] == 0 and route[-1] == 0:
            trivial_routes.append((route, bound))
        else:
            non_trivial_routes.append((route, bound))
    
    # Sort non-trivial routes by bound (descending) to prioritize promising ones
    non_trivial_routes.sort(key=lambda x: x[1], reverse=True)
    
    # Combine: non-trivial first, then trivial as fallback
    all_routes = non_trivial_routes + trivial_routes
    
    routes_evaluated_count = 0
    batch_size = 50
    max_cycles = 50
    cycle_count = 0
    
    # CONTINUOUS CYCLE: Process in batches until valid solution found
    while (cycle_count < max_cycles and 
           time.time() - start_time < timeout):
        
        # Check if we already have a valid solution
        if (best_solution['capital'] != float('-inf') and 
            best_solution['capital'] >= initial_capital):
            break  # Valid solution found, exit cycle
        
        # Get next batch of unevaluated routes
        remaining_routes = [(r, b) for r, b in all_routes 
                           if tuple(r) not in evaluated_routes]
        
        if not remaining_routes:
            break  # No more routes to evaluate
        
        # Sort remaining routes by bound (descending) to prioritize promising ones
        remaining_routes.sort(key=lambda x: x[1], reverse=True)
        
        # Process next batch (up to batch_size routes)
        batch_routes = remaining_routes[:batch_size]
        routes_in_batch = 0
        
        for route, bound in batch_routes:
            if time.time() - start_time > timeout:
                stats['timeout'] = True
                break
            
            # Quick feasibility check: skip routes that can't maintain initial capital
            if bound < initial_capital:
                continue  # Skip obviously infeasible routes
            
            # Adaptive pruning: skip if bound is too low compared to current best
            if best_solution['capital'] != float('-inf'):
                if bound < best_solution['capital'] - bound_tolerance:
                    continue
            
            route_tuple = tuple(route)
            
            # Check if already evaluated (from cache/memory) - reuse cached result
            if route_tuple in evaluated_routes:
                # Reuse memorized result from previous cycle
                max_capital, decisions = evaluated_routes[route_tuple]
            else:
                # Evaluate with DP and store in cache for future reuse
                routes_evaluated_count += 1
                routes_in_batch += 1
                stats['routes_evaluated'] += 1
                max_capital, decisions = solve_transactions_dp(route, instance)
                
                # Store in cache/memory for reuse in future cycles
                evaluated_routes[route_tuple] = (max_capital, decisions)
            
            # Update best solution if better
            if max_capital is not None:
                route_scores.append((route, max_capital))
                if max_capital > best_solution['capital']:
                    best_solution['capital'] = max_capital
                    best_solution['route'] = route
                    best_solution['decisions'] = decisions
                
                # If valid solution found, exit cycle early
                if max_capital >= initial_capital:
                    break
        
        cycle_count += 1
        
        # If no routes were evaluated in this batch (all skipped or already cached), break
        if routes_in_batch == 0:
            break
    
    # Round 2: Generate additional routes based on successful patterns
    # CRITICAL IMPROVEMENT: Skip Round 2 for large instances (n >= 15) to eliminate cycling
    # If we have good solutions, try variations of successful routes
    if n < 15 and route_scores and time.time() - start_time < timeout * 0.8:
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
    # CRITICAL IMPROVEMENT: Skip safety check for large instances (n >= 15) to avoid cycling
    final_capital = best_solution['capital'] if best_solution['capital'] != float('-inf') else None
    no_valid_solution = (final_capital is None or final_capital < instance['initial_capital'])
    
    if n < 15 and no_valid_solution and time.time() - start_time < timeout * 0.95:
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

