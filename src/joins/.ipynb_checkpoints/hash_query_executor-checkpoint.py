import pandas as pd
from typing import List, Dict, Callable
import time
from joins.hash_joins import (
    simple_hash_join, 
    grace_hash_join, 
    hybrid_hash_join, 
    radix_hash_join
)

JOIN_ALGORITHMS: Dict[str, Callable] = {
    "simple": simple_hash_join,
    "grace": grace_hash_join,
    "hybrid": hybrid_hash_join,
    "radix": radix_hash_join,
    "pandas": pd.merge,
}

def execute_query_plan(
    tables: Dict[str, pd.DataFrame], 
    plan: List[Dict], 
    join_algorithm: str = "simple") -> pd.DataFrame:
    """
    Executes a query plan step-by-step, capturing execution time, result metrics, 
    and detailed partition skew diagnostics.
    """
    join_func = JOIN_ALGORITHMS.get(join_algorithm)
    if not join_func:
        raise ValueError(f"Unknown algorithm: {join_algorithm}")

    result_df = tables[plan[0]['left']].copy()
    
    for step in plan:
        right_table_name = step['right']
        join_key = step['on']
        
        print(f"\n>>> Joining with '{right_table_name}' on key '{join_key}' using '{join_algorithm}'...")
        
        right_df = tables[right_table_name]
        
        # --- Execute and Capture Diagnostics ---
        start_time = time.time()
        
        if join_func == pd.merge:
            result_df = join_func(
                result_df, right_df, 
                on=join_key, how='inner', 
                suffixes=('_left', '_right')
            )
            diagnostics = {"info": "Using pandas internal implementation."}
        else:
            result_df, diagnostics = join_func(
                result_df, right_df, 
                key=join_key, 
                lsuffix='_left', 
                rsuffix='_right'
            )
            
        end_time = time.time()
        
        print(f"    Execution time: {end_time - start_time:.4f} seconds")
        print(f"    Result size: {len(result_df)} rows")
        
        if "df1_partition_sizes" in diagnostics:
            print("    --- Partition Skew Report ---")
            print("    Build Table Partition Sizes:", diagnostics["df1_partition_sizes"])
            print("    Probe Table Partition Sizes:", diagnostics["df2_partition_sizes"])

    return result_df