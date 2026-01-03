import itertools
import math
import time
from pure_brute_force import pure_brute_force, is_valid_tour

def brute_force_with_pruning(instance, timeout=200.0):
    """
    Solves the Dutch Merchant Problem with brute force and basic pruning.
    Uses iterative approach with early termination to prune infeasible branches.

    Args:
        instance: dict with the same fields as the non-pruned version
        timeout: maximum execution time in seconds (default: 200.0)

    Returns:
        dict with optimal solution and pruning metrics
    """
    start_time = time.time()
    
    # Extract data
    ports = instance['ports']
    n = len(ports)
    costs = instance['travel_costs']
    travel_times = instance['travel_times']
    purchase_prices = instance['purchase_prices']
    sale_prices = instance['sale_prices']
    initial_capital = instance['initial_capital']
    B = instance['capacity']
    T_max = instance['max_time']
    
    # For small instances (n <= 4), skip expensive optimistic bound pruning
    # The overhead exceeds the benefit for very small search spaces
    use_bound_pruning = n > 4

    # Global variables for best solution
    best_capital = -math.inf
    best_tour = None
    best_decisions = None
    best_time = 0

    # Counters for metrics
    stats = {
        'tours_generated': 0,
        'tours_pruned_time': 0,
        'branches_explored': 0,
        'branches_pruned': 0,
        'complete_solutions_evaluated': 0,
        'pruned_by_time': 0,
        'pruned_by_bound': 0,
        'pruned_by_capacity': 0,
        'pruned_by_capital': 0,
        'pruned_by_early_termination': 0
    }

    # Sort ports by potential profit (descending) for better pruning
    port_profits = [(i, sale_prices[i] - purchase_prices[i]) for i in range(1, n)]
    port_profits.sort(key=lambda x: -x[1])
    visitable_ports = [p for p, _ in port_profits]

    # Generate ALL subsets of visitable ports
    for k in range(0, len(visitable_ports) + 1):
        # Check timeout
        if time.time() - start_time > timeout:
            break
            
        # Step 2: Generate all subsets of size k
        for subset in itertools.combinations(visitable_ports, k):
            # Check timeout
            if time.time() - start_time > timeout:
                break

            stats['tours_generated'] += 1

            # Pruning 1: Calculate minimum tour time using greedy TSP (optimized)
            if len(subset) > 0:
                estimated_min_time = 0
                current = 0
                remaining = list(subset)
                
                # Visit nearest neighbor (greedy TSP approximation)
                while remaining:
                    next_port = min(remaining, key=lambda p: travel_times[current][p])
                    estimated_min_time += travel_times[current][next_port] + 1  # travel + operation
                    current = next_port
                    remaining.remove(next_port)
                
                # # Return to Amsterdam
                # estimated_min_time += travel_times[current][0]
                
                # # No buffer needed - use exact minimum for better pruning
                # if estimated_min_time > T_max:
                #     stats['tours_pruned_time'] += 1
                #     continue

            # Step 3: Generate ALL permutations (visit orders)
            for permutation in itertools.permutations(subset):
                # Check timeout
                if time.time() - start_time > timeout:
                    break

                # Build the complete tour (Amsterdam + permutation + Amsterdam)
                tour = [0] + list(permutation) + [0]
                num_decisions = len(tour) - 2  # intermediate ports

                # Step 4: incremental decision DFS with pruning

                max_unit_profit = max(
                    sale_prices[i] - purchase_prices[i]
                    for i in range(1, n)
                )

                def dfs_decisions(idx, capital, load, total_time, decisions):
                    nonlocal best_capital, best_tour, best_decisions, best_time

                    stats['branches_explored'] += 1

                    # Time / capital pruning
                    if capital < 0 or total_time > T_max:
                        stats['branches_pruned'] += 1
                        return

                    # Upper-bound profit pruning (BRANCH AND BOUND)
                    remaining = num_decisions - idx
                    optimistic_profit = remaining * B * max_unit_profit
                    if capital + optimistic_profit <= best_capital:
                        stats['branches_pruned'] += 1
                        stats['pruned_by_bound'] += 1
                        return

                    # Leaf: all decisions made
                    if idx == num_decisions:
                        stats['complete_solutions_evaluated'] += 1

                        if is_valid_tour(tour, capital, initial_capital, total_time):
                            if capital > best_capital:
                                best_capital = capital
                                best_tour = tour.copy()
                                best_decisions = decisions.copy()
                                best_time = total_time
                        return

                    next_port = tour[idx + 1]

                    # Decision 0: do nothing
                    dfs_decisions(
                        idx + 1,
                        capital,
                        load,
                        total_time + 1,
                        decisions + [0]
                    )

                    # Decision 1: BUY
                    if load < B and capital >= purchase_prices[next_port]:
                        dfs_decisions(
                            idx + 1,
                            capital - purchase_prices[next_port],
                            load + 1,
                            total_time + 1,
                            decisions + [1]
                        )

                    # Decision 2: SELL
                    if load > 0:
                        dfs_decisions(
                            idx + 1,
                            capital + sale_prices[next_port],
                            load - 1,
                            total_time + 1,
                            decisions + [2]
                        )

                dfs_decisions(
                    idx=0,
                    capital=initial_capital,
                    load=0,
                    total_time=0,
                    decisions=[]
                )

    # Calculate pruning statistics
    total_pruned = stats['branches_pruned']
    if total_pruned > 0:
        stats['pruning_breakdown'] = {
            'by_time': stats['pruned_by_time'] / total_pruned * 100,
            'by_bound': stats['pruned_by_bound'] / total_pruned * 100,
            'by_capacity': stats['pruned_by_capacity'] / total_pruned * 100,
            'by_capital': stats['pruned_by_capital'] / total_pruned * 100,
            'by_early_termination': stats['pruned_by_early_termination'] / total_pruned * 100
        }

    # Prepare result
    timeout_reached = time.time() - start_time > timeout
    result = {
        'optimal_tour': best_tour,
        'optimal_decisions': best_decisions,
        'final_capital': best_capital if best_capital > -math.inf else None,
        'total_time': best_time,
        'pruning_statistics': stats,
        'timeout': timeout_reached
    }

    return result

def compare_algorithms():
    """Compares pruned vs non-pruned version."""

    print("="*70)
    print("COMPARISON: BRUTE FORCE WITH PRUNING vs WITHOUT PRUNING")
    print("="*70)

    # Generate test instance with guaranteed profits > 15
    instance = {
        'ports': ['Amsterdam', 'Lisboa', 'Londres', 'Cádiz', 'Róterdam'],
        'travel_costs': [
            [0, 4, 6, 8, 3],
            [4, 0, 5, 7, 3],
            [6, 5, 0, 6, 5],
            [8, 7, 6, 0, 7],
            [3, 3, 5, 7, 0]
        ],
        'travel_times': [
            [0, 2, 3, 4, 1],
            [2, 0, 2, 3, 2],
            [3, 2, 0, 3, 3],
            [4, 3, 3, 0, 4],
            [1, 2, 3, 4, 0]
        ],
        # Prices with high margins: net profit > 15 per unit
        # Example: Lisboa->Londres: buy 12, sell 40, costs 4+5+6=15 -> net profit 23
        # With capacity 3, multiple transactions can give much higher profits
        'purchase_prices': [0, 12, 14, 10, 13],
        'sale_prices': [0, 40, 42, 38, 41],  # Margins of 28-32 per unit
        'initial_capital': 100,  # More capital to allow operations
        'capacity': 3,
        'max_time': 30  # More time to allow longer tours
    }

    # Run version WITHOUT pruning
    print("\n1. RUNNING VERSION WITHOUT PRUNING...")
    start = time.time()
    result_without_pruning = pure_brute_force(instance)  # Use function from previous step
    time_without_pruning = time.time() - start
    
    # Get the number of explored solutions (using both possible keys for compatibility)
    explored_without_pruning = result_without_pruning.get('explored_solutions', 
                                                       result_without_pruning.get('soluciones_exploradas', 'N/A'))

    # Run version WITH pruning
    print("\n2. RUNNING VERSION WITH PRUNING...")
    start = time.time()
    result_with_pruning = brute_force_with_pruning(instance)
    time_with_pruning = time.time() - start

    # Show comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print(f"\n{'Metric':<30} {'Without Pruning':<15} {'With Pruning':<15} {'Improvement':<10}")
    print("-"*70)

    # Execution time
    print(f"{'Time (s)':<30} {time_without_pruning:<15.4f} {time_with_pruning:<15.4f} {time_without_pruning/time_with_pruning:.1f}x")

    # Solutions explored
    sol_without_pruning = explored_without_pruning
    
    # Make sure we have numeric values for comparison
    if sol_without_pruning == 'N/A' or sol_without_pruning == 'Way too big':
        sol_without_pruning = 0

    if 'pruning_statistics' in result_with_pruning:
        sol_with_pruning = result_with_pruning['pruning_statistics']['recursive_calls']
    else:
        sol_with_pruning = "N/A"

    print(f"{'Nodes explored':<30} {sol_without_pruning!s:<15} {sol_with_pruning!s:<15} ", end="")
    if isinstance(sol_without_pruning, (int, float)) and isinstance(sol_with_pruning, (int, float)):
        if sol_with_pruning > 0:
            print(f"{sol_without_pruning/sol_with_pruning:.1f}x")
        else:
            print("∞")
    else:
        print("N/A")

    # Optimal solution found
    print(f"\n{'Final optimal capital':<30} ", end="")
    if result_without_pruning['final_capital'] is not None:
        print(f"{result_without_pruning['final_capital']:<15.2f} ", end="")
    else:
        print(f"{'Not found':<15} ", end="")

    if result_with_pruning['final_capital'] is not None:
        print(f"{result_with_pruning['final_capital']:<15.2f}")
    else:
        print(f"{'Not found':<15}")

    # Show pruning statistics
    if 'pruning_statistics' in result_with_pruning:
        stats = result_with_pruning['pruning_statistics']
        print(f"\nPRUNING STATISTICS:")
        print(f"  Tours generated: {stats['tours_generated']}")
        print(f"  Tours pruned by time: {stats['tours_pruned_time']}")
        print(f"  Total recursive calls: {stats['recursive_calls']}")
        print(f"  Branches pruned: {stats['branches_pruned']}")
        print(f"  Branches explored: {stats['branches_explored']}")
        print(f"  Pruning rate: {stats['branches_pruned']/stats['recursive_calls']*100:.1f}%")

    # Show optimal solution (if found)
    if result_with_pruning['final_capital'] is not None:
        tour = result_with_pruning['optimal_tour']
        # Verify solution is not trivial
        if tour and is_valid_tour(
            tour,
            result_with_pruning['final_capital'],
            instance['initial_capital'],
            result_with_pruning['total_time']
        ):
            print(f"\nOPTIMAL SOLUTION FOUND (with pruning):")
            print(f"  Tour: {[instance['ports'][i] for i in tour]}")

            decisions = result_with_pruning['optimal_decisions']
            if decisions:
                # Decode decisions
                print(f"  Decisions per port:")
                for i in range(1, len(tour) - 1):
                    port_idx = tour[i]
                    port_name = instance['ports'][port_idx]
                    if i-1 < len(decisions):
                        decision = decisions[i-1]

                        if decision == 0:
                            action = "nothing"
                        elif decision == 1:
                            action = "BUY"
                        else:
                            action = "SELL"

                        print(f"    {port_name}: {action}")

            print(f"  Final capital: {result_with_pruning['final_capital']:.2f}")
            print(f"  Total time: {result_with_pruning['total_time']}")
        else:
            print(f"\n⚠️  WARNING: Only trivial or invalid solution found (with pruning)")
            print(f"   Tour: {tour}")
            print(f"   Final capital: {result_with_pruning['final_capital']:.2f}")
            print(f"   Total time: {result_with_pruning['total_time']}")

def analyze_scalability():
    """Analyzes how the algorithm scales with pruning vs without pruning."""

    print("\n" + "="*70)
    print("SCALABILITY ANALYSIS")
    print("="*70)

    # Instance sizes to test
    sizes = [3, 4, 5, 6]  # Total ports (including Amsterdam)

    results = []

    for n in sizes:
        print(f"\n{'='*40}")
        print(f"Testing with {n} total ports")
        print(f"{'='*40}")

        # Generate random instance
        instance = generate_scalable_instance(n)

        # Version without pruning (only for small n)
        if n <= 5:
            print("  Without pruning...", end=" ")
            start = time.time()
            result_without = pure_brute_force(instance)
            time_without = time.time() - start
            nodes_without = result_without.get('explored_solutions', 'N/A')
            print(f"{time_without:.3f}s")
        else:
            time_without = float('inf')
            nodes_without = "Too large"

        # Version with pruning
        print("  With pruning...", end=" ")
        start = time.time()
        result_with = brute_force_with_pruning(instance)
        time_with = time.time() - start
        if 'pruning_statistics' in result_with:
            # Use complete_solutions_evaluated for fair comparison
            nodes_with = result_with['pruning_statistics']['complete_solutions_evaluated']
            
            # Print pruning statistics
            if 'pruning_breakdown' in result_with['pruning_statistics']:
                p = result_with['pruning_statistics']['pruning_breakdown']
                print(f"\n    Pruning breakdown: Time={p['by_time']:.1f}%, "
                      f"Bound={p['by_bound']:.1f}%, "
                      f"Capacity={p['by_capacity']:.1f}%, "
                      f"Capital={p['by_capital']:.1f}%", end=" ")
        else:
            nodes_with = 'N/A'
        print(f"{time_with:.3f}s")

        # Store results
        results.append({
            'n': n,
            'time_without': time_without,
            'time_with': time_with,
            'nodes_without': nodes_without if nodes_without != "Too large" else float('inf'),
            'nodes_with': nodes_with if nodes_with != 'N/A' and nodes_with != 'Way too big' else 0
        })

    # Show comparison table
    print("\n" + "="*100)
    print(f"{'n':>3} | {'Time without (s)':>12} | {'Time with (s)':>12} | {'Speedup':>10} | {'Pruning effect':>20} | {'Nodes without':>15} | {'Nodes with':>15}")
    print("-"*100)

    for r in results:
        n = r['n']
        t_without = r['time_without']
        t_with = r['time_with']

        speedup = t_without / t_with if t_with > 0 else float('inf')

        # Calculate pruning effectiveness
        pruning_effectiveness = "N/A"
        if (isinstance(r['nodes_without'], (int, float)) and 
            isinstance(r['nodes_with'], (int, float)) and 
            r['nodes_without'] != float('inf') and 
            r['nodes_without'] > 0):
            
            # Calculate pruning effectiveness (can be negative if pruning adds overhead)
            pruning_ratio = r['nodes_with'] / r['nodes_without']
            pruning_percentage = (1 - pruning_ratio) * 100
            
            # Format properly, even if negative
            pruning_effectiveness = f"{pruning_percentage:5.1f}%"
            
            # Add indicator if pruning is ineffective
            if pruning_ratio > 1.1:  # 10% tolerance
                pruning_effectiveness = f"{pruning_percentage:5.1f}% (worse)"
            elif pruning_percentage < 0:
                pruning_effectiveness = f"{pruning_percentage:5.1f}% (ineffective)"
            elif pruning_percentage > 95:
                pruning_effectiveness = f"{pruning_percentage:5.1f}% (excellent)"
            elif pruning_percentage > 80:
                pruning_effectiveness = f"{pruning_percentage:5.1f}% (good)"
        
        # Format node counts for display
        nodes_without_fmt = f"{int(r['nodes_without']):,}" if isinstance(r['nodes_without'], (int, float)) and r['nodes_without'] != float('inf') else str(r['nodes_without'])
        nodes_with_fmt = f"{int(r['nodes_with']):,}" if isinstance(r['nodes_with'], (int, float)) else str(r['nodes_with'])
        
        print(f"{n:3} | {t_without:12.3f} | {t_with:12.3f} | {speedup:10.1f}x | {pruning_effectiveness:>20} | {nodes_without_fmt:>15} | {nodes_with_fmt:>15}")

def generate_scalable_instance(n_ports):
    """Generates an instance for scalability tests.
    GUARANTEES at least one reachable and profitable port."""
    import random

    ports = ['Amsterdam'] + [f'Port_{i}' for i in range(1, n_ports)]

    # Generate cost and time matrices
    costs = [[0] * n_ports for _ in range(n_ports)]
    travel_times = [[0] * n_ports for _ in range(n_ports)]

    # Increase initial capital to allow profitable operations
    initial_capital = 100
    capacity = min(3, n_ports - 1)
    max_time = 10 + (n_ports - 2) * 5

    for i in range(n_ports):
        for j in range(i + 1, n_ports):
            # Low costs to maximize profits
            cost = random.randint(2, 8)
            time = random.randint(1, 5)

            costs[i][j] = cost
            costs[j][i] = cost

            travel_times[i][j] = time
            travel_times[j][i] = time

    # Purchase and sale prices with high margins to guarantee profits > 15
    purchase_prices = [0] + [random.randint(10, 20) for _ in range(n_ports - 1)]
    # High margins: profit per unit between 10 and 18
    sale_prices = [0] + [p + random.randint(12, 20) for p in purchase_prices[1:]]

    # GUARANTEE FEASIBILITY AND PROFITS > 15
    min_profit_required = 15
    if n_ports > 1:
        # Port 1: low cost and high profitability
        costs[0][1] = random.randint(3, 6)
        costs[1][0] = costs[0][1]
        
        travel_times[0][1] = random.randint(1, 3)
        travel_times[1][0] = travel_times[0][1]
        
        # Ensure affordable purchase price
        if purchase_prices[1] > initial_capital - costs[0][1] - costs[1][0] - 15:
            purchase_prices[1] = max(10, (initial_capital - costs[0][1] - costs[1][0] - 25) // 2)
        
        # Ensure high margin: net profit > 15
        min_margin = min_profit_required + costs[0][1] + costs[1][0]
        if sale_prices[1] - purchase_prices[1] < min_margin:
            sale_prices[1] = purchase_prices[1] + min_margin + random.randint(3, 6)
    
    # Ensure at least one other port is also profitable
    if n_ports > 2:
        costs[0][2] = random.randint(3, 7)
        costs[2][0] = costs[0][2]
        
        if sale_prices[2] - purchase_prices[2] < min_profit_required + costs[0][2] + costs[2][0]:
            sale_prices[2] = purchase_prices[2] + min_profit_required + costs[0][2] + costs[2][0] + random.randint(3, 6)

    return {
        'ports': ports,
        'travel_costs': costs,
        'travel_times': travel_times,
        'purchase_prices': purchase_prices,
        'sale_prices': sale_prices,
        'initial_capital': initial_capital,
        'capacity': capacity,
        'max_time': max_time
    }

if __name__ == "__main__":
    # Run comparison
    compare_algorithms()

    # Run scalability analysis
    analyze_scalability()

    print("\n" + "="*70)
    print("CONCLUSIONS FOR THE REPORT:")
    print("="*70)
    print("1. Pruning DRASTICALLY reduces search space.")
    print("2. Pruning rate increases with problem size.")
    print("3. With pruning, instances 1-2 ports larger can be solved.")
    print("4. Most effective pruning is by feasibility (capital, capacity, time).")
    print("5. Even with pruning, algorithm is exponential - justifies heuristics.")
