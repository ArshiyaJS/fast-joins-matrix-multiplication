# In src/query_executor.py
import pandas as pd
from typing import List, Dict, Callable
import time

# 1. --- Import all the physical operators ---
# It imports the actual functions that know how to join TWO tables.
from src.baselines.hash_joins_collision_domainSkewPartition import (
    simple_hash_join, grace_hash_join, hybrid_hash_join, radix_hash_join
)
from src.baselines.sort_merge_join import sort_merge_join

# 2. --- Create a "Registry" of Algorithms ---
# This dictionary acts as a simple, powerful "switch" statement.
# It maps a human-readable name to the actual function to be called.
# This allows you to easily add new algorithms without changing the main logic.
JOIN_ALGORITHMS: Dict[str, Callable] = {
    "simple_hash": simple_hash_join,
    "grace_hash": grace_hash_join,
    "hybrid_hash": hybrid_hash_join,
    "radix_hash": radix_hash_join,
    "sort_merge": sort_merge_join,
    "pandas": pd.merge, # A gold-standard baseline
}

# 3. --- The Main Executor Function ---
def execute_query_plan(
    tables: Dict[str, pd.DataFrame], 
    plan: List[Dict], 
    join_algorithm: str = "simple_hash"
) -> pd.DataFrame:
    """
    Executes a query plan for multi-table joins. It translates a logical plan
    into a sequence of physical join operations.
    """
    # Select the correct join function from the registry based on the user's choice.
    join_func = JOIN_ALGORITHMS.get(join_algorithm)
    if not join_func:
        raise ValueError(f"Unknown algorithm: {join_algorithm}. Available: {list(JOIN_ALGORITHMS.keys())}")

    # The starting point is always the first table mentioned in the plan.
    result_df = tables[plan[0]['left']].copy()
    
    # 4. --- Loop Through the Plan Steps ---
    # It iterates through a list of join steps, executing them one by one.
    for step in plan:
        right_table_name = step['right']
        join_key = step['on']
        
        print(f"\n>>> Joining with '{right_table_name}' on key '{join_key}' using '{join_algorithm}'...")
        
        right_df = tables[right_table_name]
        
        # --- Execute the chosen physical operator ---
        start_time = time.time()
        
        if join_func == pd.merge:
            result_df = join_func(result_df, right_df, on=join_key, how='inner', suffixes=('_left', '_right'))
            diagnostics = {"info": "Using pandas internal implementation."}
        else:
            # The intermediate result from the previous step becomes the 'left' table for the current step.
            result_df, diagnostics = join_func(result_df, right_df, key=join_key, lsuffix='_left', rsuffix='_right')
            
        end_time = time.time()
        
        # --- Report Metrics ---
        # It prints the performance and diagnostic info for this specific step.
        print(f"    Execution time: {end_time - start_time:.4f} seconds")
        print(f"    Result size: {len(result_df)} rows")
        
        if "df1_partition_sizes" in diagnostics:
            print("    --- Partition Skew Report ---")
            # ... (printing logic)

    return result_df
