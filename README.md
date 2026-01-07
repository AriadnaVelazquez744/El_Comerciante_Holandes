# The Dutch Merchant - NP Problem Solutions

This project implements different approaches to solve the Dutch Merchant Problem, a combinatorial optimization challenge that aims to maximize the profit of a merchant traveling between ports to buy and sell goods, while respecting time and capacity constraints.

## 📋 Table of Contents

- [The Dutch Merchant - NP Problem Solutions](#the-dutch-merchant---np-problem-solutions)
  - [📋 Table of Contents](#-table-of-contents)
  - [📝 Problem Description](#-problem-description)
    - [Key Parameters](#key-parameters)
    - [Main Constraints](#main-constraints)
    - [Objective](#objective)
  - [📁 Project Structure](#-project-structure)
  - [🚀 Implemented Algorithms](#-implemented-algorithms)
  - [🏗️ Instance Generation](#️-instance-generation)
  - [📊 Analysis and Results](#-analysis-and-results)
    - [BF\_graphics.ipynb](#bf_graphicsipynb)
    - [Complete\_graphics.ipynb](#complete_graphicsipynb)
  - [📦 Requirements and Installation](#-requirements-and-installation)
  - [🚀 Usage](#-usage)
    - [Instance Types](#instance-types)
    - [Basic Usage](#basic-usage)
      - [1. Pure Brute Force](#1-pure-brute-force)
      - [2. Brute Force with Pruning](#2-brute-force-with-pruning)
      - [3. Hybrid Brute Force + Dynamic Programming](#3-hybrid-brute-force--dynamic-programming)
      - [4. Efficient Solution (Two-Phase Hybrid)](#4-efficient-solution-two-phase-hybrid)
  - [📚 Documentation](#-documentation)

## 📝 Problem Description

The Dutch Merchant Problem is a combinatorial optimization challenge inspired by the operations of the historic Dutch East India Company. The goal is to plan the optimal trade route for a merchant ship that must:

- Depart from Amsterdam (port 0) and return to the same port
- Visit multiple ports (without repetition in the same journey)
- Buy and sell goods at each port
- Maximize final profit while respecting time and capacity constraints

### Key Parameters

- `n`: Total number of ports (including Amsterdam)
- `m`: Number of different types of goods available
- `B`: Maximum cargo capacity of the ship (in units of goods)
- `T_max`: Maximum allowed time for the complete route
- `r`: Initial capital available

### Main Constraints

1. **Cargo Capacity**: The ship cannot carry more than `B` units of goods
2. **Time Limit**: Total journey duration cannot exceed `T_max`
3. **Budget**: Capital cannot be negative at any point
4. **No Repetition**: The same port cannot be visited twice in the same route

### Objective

Find the sequence of ports to visit and the buy/sell operations at each one that maximizes the final capital when returning to Amsterdam, while respecting all constraints.

## 📁 Project Structure

```txt
.
├── data/                   # Generated test data
├── figures/                # Result figures and graphs
├── Documentation/          # Detailed documentation
├── BF_DP_hybrid.py         # Hybrid solution (Brute Force + Dynamic Programming)
├── BF_graphics.ipynb       # Brute Force visualization notebook
├── Complete_graphics.ipynb # Complete analysis and visualization notebook
├── comparison_test.py      # Algorithm comparison tests
├── efficient_solution.py   # Efficient solution to the problem
├── instance_generator.py   # Test instance generator
├── pruning_brute_force.py  # Pruned brute force
├── pure_brute_force.py     # Pure brute force
└── pyproject.toml          # Project dependencies
```

## 🚀 Implemented Algorithms

1. **Pure Brute Force** (`pure_brute_force.py`)
   - Explores all possible routes
   - Only suitable for very small instances (n ≤ 8 with m = 1)

2. **Pruned Brute Force** (`pruning_brute_force.py`)
   - Includes branch pruning for non-promising paths
   - Improves performance over the pure version

3. **Brute Force/Dynamic Programming Hybrid** (`BF_DP_hybrid.py`)
   - Combines brute force with dynamic programming
   - Efficient for medium-sized instances (n ≤ 12 with m = 1)

4. **Efficient Solution** (`efficient_solution.py`)
   - Implements a more efficient algorithm
   - Designed to handle larger instances (n ≥ 15 with m ≤ 15)

## 🏗️ Instance Generation

The project provides two main methods for instance generation:

1. **Direct Generation** using [instance_generator.py](cci:7://file:///home/ari/Collage/04-Forth_Year/Preimer_Semestre/DAA/El_Comerciante_Holandes/instance_generator.py:0:0-0:0):
   - Generates unified test instances for all input sizes
   - Ensures at least one profitable route exists
   - Produces instances in the single-item format

2. **Format Conversion** using [convert_single_item_to_multi_item()](cci:1://file:///home/ari/Collage/04-Forth_Year/Preimer_Semestre/DAA/El_Comerciante_Holandes/comparison_test.py:11:0-37:5) in [comparison_test.py](cci:7://file:///home/ari/Collage/04-Forth_Year/Preimer_Semestre/DAA/El_Comerciante_Holandes/comparison_test.py:0:0-0:0):
   - Converts single-item instances to multi-item format
   - Useful for testing algorithms that require multi-item instances
   - Handles the mapping of purchase and sale prices appropriately

The instances generated guarantee at least one profitable route and maintain consistency across different algorithm implementations.

## 📊 Analysis and Results

The project includes comprehensive performance analysis and comparison of the three algorithms across different instance sizes. The analysis is documented in two main notebooks:

### BF_graphics.ipynb

- **Performance Growth Analysis**: Measures how execution time scales with problem size for each algorithm
- **Solution Quality Comparison**: Compares the optimal solutions found by different methods
- **Algorithm Efficiency**: Evaluates the number of solutions explored by each approach

### Complete_graphics.ipynb

- **Scalability Testing**: Tests all algorithms on instances from 3 to 12 ports
- **Direct Comparison**: Runs all algorithms on identical instances to compare performance
- **Efficiency Metrics**: Analyzes time complexity and solution quality trade-offs
- **Visualizations**: Generates comparative charts showing execution times and solution quality

Key findings from the analysis:

- Pure brute force becomes impractical beyond 8 ports due to exponential growth in computation time
- The hybrid approach shows significant performance improvements while maintaining solution quality
- The efficient solution demonstrates superior scalability for larger instances (n > 12)

All analysis results, including raw data and visualizations, are saved in the `data/` and `figures/` directories respectively.

## 📦 Requirements and Installation

1. Clone the repository:

   ```bash
   git clone [REPOSITORY_URL]
   cd El_Comerciante_Holandes
   ```

2. Install dependencies:

   ```bash
   pip install -e .
   ```

   or install dependencies manually from `pyproject.toml`

## 🚀 Usage

### Instance Types

The algorithms use different instance formats:

1. **Brute Force and Hybrid Algorithms**:
   - Use single-item instances generated by `instance_generator.py`
   - Format: Each port has exactly one type of good (m = 1)
   - Best for small instances (n ≤ 12)

   ```python
   from instance_generator import generate_unified_test_instance
   single_item_instance = generate_unified_test_instance(n_ports=5, seed=42)
   ```

2. **Efficient Solution**:
   - Uses multi-item instances (m ≥ 1)
   - Format: Each port can have multiple types of goods
   - Required for larger instances (n > 12)

   ```python
   from efficient_solution import generate_multi_item_instance
   
   # Generate a multi-item instance with 5 ports and 3 item types
   multi_item_instance = generate_multi_item_instance(
       n_ports=5,         # Number of ports (including Amsterdam)
       m_items=3,          # Number of item types per port (m > 1)
       k_max_units=2,      # Maximum units per buy/sell operation
       seed=42             # Random seed for reproducibility
   )
   
   # Or convert from single-item to multi-item format
   from comparison_test import convert_single_item_to_multi_item
   multi_item_instance = convert_single_item_to_multi_item(single_item_instance)
   ```

   **Multi-item Instance Structure**:

   ```python
   {
       'ports': List[str],                      # Port names
       'travel_costs': List[List[float]],       # Travel costs between ports
       'travel_times': List[List[int]],         # Travel times between ports
       'purchase_prices': Dict[int, List[float]],  # Purchase prices per port and item
       'sale_prices': Dict[int, List[float]],      # Sale prices per port and item
       'initial_capital': float,                # Starting capital
       'capacity': int,                         # Maximum cargo capacity
       'max_time': int,                         # Maximum allowed travel time
       'num_items': int,                        # Number of item types (m)
       'max_units_per_op': int                  # Maximum units per transaction
   }
   ```

### Basic Usage

1. Generate a test instance:

   ```python
   from instance_generator import generate_unified_test_instance
   from comparison_test import convert_single_item_to_multi_item
   
   # For brute force/hybrid:
   single_item_instance = generate_unified_test_instance(n_ports=5, seed=42)
   
   # For efficient solution:
   multi_item_instance = convert_single_item_to_multi_item(single_item_instance)
   ```

2. Available Solution Methods:

   #### 1. Pure Brute Force

   ```python
   from pure_brute_force import pure_brute_force
   
   # Parameters:
   # - instance: Single-item instance dictionary
   # - timeout: Maximum execution time in seconds (default: 200.0)
   result = pure_brute_force(single_item_instance, timeout=200.0)
   ```

   #### 2. Brute Force with Pruning

   ```python
   from pruning_brute_force import brute_force_with_pruning
   
   # Parameters:
   # - instance: Single-item instance dictionary
   # - timeout: Maximum execution time in seconds (default: 200.0)
   result = brute_force_with_pruning(single_item_instance, timeout=200.0)
   ```

   #### 3. Hybrid Brute Force + Dynamic Programming

   ```python
   from BF_DP_hybrid import hybrid_brute_force_dp
   
   # Parameters:
   # - instance: Single-item instance dictionary
   # - timeout: Maximum execution time in seconds (default: 200.0)
   result = hybrid_brute_force_dp(single_item_instance, timeout=200.0)
   ```

   #### 4. Efficient Solution (Two-Phase Hybrid)

   ```python
   from efficient_solution import two_phase_hybrid_solve
   
   # Parameters:
   # - instance: Multi-item instance dictionary
   # - timeout: Maximum execution time in seconds (default: 200.0)
   result = two_phase_hybrid_solve(multi_item_instance, timeout=200.0)
   ```

3. Common Return Structure:
   All methods return a dictionary with the following structure:

   ```python
   {
       'optimal_tour': list,       # List of port indices in visit order
       'optimal_decisions': list,  # List of buy/sell decisions at each port
       'final_capital': float,     # Final profit after completing the tour
       'total_time': float,        # Execution time in seconds
       'explored_solutions': int,  # Number of solutions evaluated
       'timeout': bool            # Whether the algorithm timed out
   }
   ```

4. Example Usage:

   ```python
   # Example using the hybrid approach
   from BF_DP_hybrid import hybrid_brute_force_dp
   
   result = hybrid_brute_force_dp(single_item_instance)
   if not result['timeout']:
       print(f"Best route: {result['optimal_tour']}")
       print(f"Final profit: {result['final_capital']}")
       print(f"Explored {result['explored_solutions']} solutions in {result['total_time']:.2f}s")
   else:
       print("Search timed out before finding optimal solution")
   ```

> **Note about comparison_test.py**:
> The comparison test is primarily for internal validation and doesn't provide meaningful performance comparisons between algorithms due to their different instance requirements and optimization goals. For accurate comparisons, please refer to the analysis in the Jupyter notebooks.

## 📚 Documentation

Detailed documentation is available in the `Documentation/` directory:

- `Complete_Report.pdf`: Comprehensive project report covering all aspects including problem definition, algorithmic approaches, complexity analysis, experimental results, and conclusions. Includes comparisons between all implemented solutions.

- `Brute_Force_Inform.pdf`: Detailed analysis of brute force approaches, including:
  - Pure brute force implementation
  - Brute force with pruning optimizations
  - Hybrid brute force with dynamic programming
  - Performance comparisons and scalability analysis

- `Efficient_Solution_Report.pdf`: Documentation of the two-phase hybrid solution, featuring:
  - Route generation using beam search
  - Transaction optimization via dynamic programming
  - Adaptive strategies for large instances
  - Performance evaluation and optimization techniques

- `Reducción.pdf`/`Reduction.pdf`: Theoretical analysis proving the NP-Completeness of the Dutch Merchant Problem through polynomial-time reduction from the Prize-Collecting TSP (PCTSP).
