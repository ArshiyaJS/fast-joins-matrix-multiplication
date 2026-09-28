import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Tuple, Any
import time

# 1. --- Import all physical pairwise operators ---
from src.baselines.hash_joins_collision_domainSkewPartition import (
    simple_hash_join, grace_hash_join, hybrid_hash_join, radix_hash_join
)
from src.baselines.sort_merge_join import sort_merge_join

# --- Import the Leapfrog Triejoin Engine ---
# Assuming leapfrog_generalised.py is placed in src/ or accessible in python path
from src.baselines.leapfrog_generalised import RobustGeneralLFTJEngine


# 2. --- Create a "Registry" of Algorithms ---
JOIN_ALGORITHMS: Dict[str, Callable] = {
    "simple_hash": simple_hash_join,
    "grace_hash": grace_hash_join,
    "hybrid_hash": hybrid_hash_join,
    "radix_hash": radix_hash_join,
    "sort_merge": sort_merge_join,
    "pandas": pd.merge,  # Gold-standard baseline
    "leapfrog": None,    # Handled specially as a multi-way join engine
}


# 3. --- The Main Executor Function ---
def execute_query_plan(
    tables: Dict[str, pd.DataFrame], 
    plan: List[Dict], 
    join_algorithm: str = "simple_hash"
) -> pd.DataFrame:
    """
    Executes a query plan for multi-table joins. Supports both traditional 
    pairwise physical operators and multi-way Leapfrog Triejoins (LFTJ).
    """
    if join_algorithm not in JOIN_ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {join_algorithm}. Available: {list(JOIN_ALGORITHMS.keys())}")

    # --- Special Handling for Multi-Way Leapfrog Triejoin ---
    if join_algorithm == "leapfrog":
        print(f"\n>>> Executing multi-way join using Leapfrog Triejoin (LFTJ)...")
        start_time = time.time()
        
        # Automatically construct the multi-way query definition from all provided tables
        # Each table name, dataframe, and its column attributes form the relation schema.
        query_def = [
            {
                'name': name, 
                'table': df, 
                'attrs': list(df.columns)
            }
            for name, df in tables.items()
        ]
        
        # Instantiate and execute the robust general LFTJ engine
        engine = RobustGeneralLFTJEngine(query_def)
        result_df, diagnostics = engine.execute()
        
        end_time = time.time()
        
        # Report Metrics
        print(f"    Execution time: {end_time - start_time:.4f} seconds")
        print(f"    Result size: {len(result_df)} rows")
        print(f"    Optimal Variable Order: {diagnostics.get('variable_order')}")
        
        return result_df

    # --- Standard Pairwise Join Execution Flow ---
    join_func = JOIN_ALGORITHMS.get(join_algorithm)
    
    # The starting point is always the first table mentioned in the plan.
    result_df = tables[plan[0]['left']].copy()
    
    # Loop Through the Plan Steps
    for step in plan:
        right_table_name = step['right']
        join_key = step['on']
        
        print(f"\n>>> Joining with '{right_table_name}' on key '{join_key}' using '{join_algorithm}'...")
        
        right_df = tables[right_table_name]
        
        # Execute the chosen physical operator
        start_time = time.time()
        
        if join_func == pd.merge:
            result_df = join_func(result_df, right_df, on=join_key, how='inner', suffixes=('_left', '_right'))
            diagnostics = {"info": "Using pandas internal implementation."}
        else:
            # The intermediate result from the previous step becomes the 'left' table
            result_df, diagnostics = join_func(result_df, right_df, key=join_key, lsuffix='_left', rsuffix='_right')
            
        end_time = time.time()
        
        # Report Metrics
        print(f"    Execution time: {end_time - start_time:.4f} seconds")
        print(f"    Result size: {len(result_df)} rows")
        
        if "df1_partition_sizes" in diagnostics:
            print("    --- Partition Skew Report ---")
            # ... (partition skew printing logic if applicable)

    return result_df