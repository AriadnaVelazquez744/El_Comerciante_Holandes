"""
Unified instance generator for the Dutch Merchant Problem.
Ensures consistent test instances across all algorithms and sizes.
"""
import random


def generate_unified_test_instance(n_ports, seed=42):
    """
    Generates a unified test instance that works for all sizes (n = 3 to 8).
    Ensures consistency: same seed produces same instance.
    Guarantees at least one profitable route exists.
    
    Args:
        n_ports: total number of ports (including Amsterdam)
        seed: random seed for reproducibility
        
    Returns:
        dict with the problem instance
    """
    random.seed(seed)
    
    # Port names
    ports = ['Amsterdam'] + [f'Port_{i}' for i in range(1, n_ports)]
    
    # Initialize symmetric matrices
    costs = [[0] * n_ports for _ in range(n_ports)]
    travel_times = [[0] * n_ports for _ in range(n_ports)]
    
    # Parameters that scale appropriately with instance size
    # Initial capital: enough for operations but not excessive
    initial_capital = 50 + (n_ports - 3) * 15  # 50 for n=3, scales up
    
    # Capacity: small for meaningful constraints
    capacity = min(3, n_ports - 1)
    
    # Max time: scales with number of ports
    max_time = 15 + (n_ports - 2) * 8
    
    # Generate cost and time matrices
    for i in range(n_ports):
        for j in range(i + 1, n_ports):
            # Moderate costs (2-10) to allow profitable operations
            cost = random.randint(2, 10)
            time_val = random.randint(1, 6)
            
            costs[i][j] = cost
            costs[j][i] = cost
            travel_times[i][j] = time_val
            travel_times[j][i] = time_val
    
    # Purchase and sale prices with good profit margins
    # Purchase prices: moderate (8-20)
    purchase_prices = [0] + [random.randint(8, 20) for _ in range(n_ports - 1)]
    
    # Sale prices: high margins (purchase + 12-25)
    sale_prices = [0] + [p + random.randint(12, 25) for p in purchase_prices[1:]]
    
    # GUARANTEE FEASIBILITY: Ensure at least one profitable route exists
    minimum_required_profit = 15
    
    if n_ports > 1:
        # Port 1: make it highly profitable and accessible
        costs[0][1] = random.randint(3, 7)  # Low travel cost
        costs[1][0] = costs[0][1]
        
        travel_times[0][1] = random.randint(1, 4)  # Low travel time
        travel_times[1][0] = travel_times[0][1]
        
        # Ensure purchase price is affordable
        if purchase_prices[1] > initial_capital - costs[0][1] - costs[1][0] - 15:
            purchase_prices[1] = max(8, (initial_capital - costs[0][1] - costs[1][0] - 20) // 2)
        
        # Ensure high margin: net profit > 15 per unit
        min_margin = minimum_required_profit + costs[0][1] + costs[1][0]
        if sale_prices[1] - purchase_prices[1] < min_margin:
            sale_prices[1] = purchase_prices[1] + min_margin + random.randint(3, 8)
    
    # Ensure at least one more port is profitable (if n_ports > 2)
    if n_ports > 2:
        costs[0][2] = random.randint(3, 8)
        costs[2][0] = costs[0][2]
        
        travel_times[0][2] = random.randint(1, 5)
        travel_times[2][0] = travel_times[0][2]
        
        # Ensure port 2 is also profitable
        if sale_prices[2] - purchase_prices[2] < minimum_required_profit + costs[0][2] + costs[2][0]:
            sale_prices[2] = purchase_prices[2] + minimum_required_profit + costs[0][2] + costs[2][0] + random.randint(3, 8)
    
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




