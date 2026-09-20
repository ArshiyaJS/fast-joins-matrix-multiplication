import pandas as pd
from typing import List, Dict

# --------------------------------------------------------------------------
# 1. Simple Hash Join (Classic In-Memory Implementation)
# --------------------------------------------------------------------------
def simple_hash_join(df1: pd.DataFrame, df2: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Performs a classic in-memory hash join.
    - It selects the smaller DataFrame to build the hash table.
    - It correctly handles many-to-many joins (duplicate keys).
    - It correctly ignores NULL keys.
    """
    if len(df1) <= len(df2):
        build_df, probe_df = df1, df2
    else:
        build_df, probe_df = df2, df1

    # --- Build Phase ---
    hash_table: Dict[object, List[tuple]] = {}
    build_cols = build_df.columns
    
    for row in build_df.itertuples(index=False, name=None):
        row_dict = dict(zip(build_cols, row))
        key_val = row_dict.get(key)

        if pd.isna(key_val):
            continue
        
        if key_val not in hash_table:
            hash_table[key_val] = []
        
        hash_table[key_val].append(row)

    # --- Probe Phase ---
    joined_rows = []
    probe_cols = probe_df.columns
    
    for probe_row_tuple in probe_df.itertuples(index=False, name=None):
        probe_row_dict = dict(zip(probe_cols, probe_row_tuple))
        key_val = probe_row_dict.get(key)
        
        if key_val in hash_table:
            for build_row_tuple in hash_table[key_val]:
                build_row_dict = dict(zip(build_cols, build_row_tuple))
                del build_row_dict[key]
                joined_row = {**probe_row_dict, **build_row_dict}
                joined_rows.append(joined_row)

    if not joined_rows:
        final_cols = list(probe_cols) + [c for c in build_cols if c != key]
        return pd.DataFrame(columns=final_cols)
        
    return pd.DataFrame(joined_rows)


# --------------------------------------------------------------------------
# 2. Grace Hash Join (Vectorized Partition-based)
# --------------------------------------------------------------------------
def grace_hash_join(df1: pd.DataFrame, df2: pd.DataFrame, key: str, num_partitions: int = 8) -> pd.DataFrame:
    """
    Performs a Grace Hash Join using vectorized Pandas operations for speed.
    """
    # Vectorized partitioning indices (ignoring NaNs)
    h_keys1 = df1[key].apply(lambda x: abs(hash(x)) % num_partitions if pd.notna(x) else -1)
    h_keys2 = df2[key].apply(lambda x: abs(hash(x)) % num_partitions if pd.notna(x) else -1)
    
    valid_df1 = df1[h_keys1 != -1]
    valid_keys1 = h_keys1[h_keys1 != -1]
    
    valid_df2 = df2[h_keys2 != -1]
    valid_keys2 = h_keys2[h_keys2 != -1]
    
    # Create dictionary mapping partition ID to subset DataFrame using Pandas groupby
    groups1 = {p: grp for p, grp in valid_df1.groupby(valid_keys1)}
    groups2 = {p: grp for p, grp in valid_df2.groupby(valid_keys2)}
    
    # --- Join Phase ---
    partition_results = []
    common_partitions = set(groups1.keys()) & set(groups2.keys())
    
    for p_idx in common_partitions:
        res = simple_hash_join(groups1[p_idx], groups2[p_idx], key)
        if not res.empty:
            partition_results.append(res)
            
    if not partition_results:
        final_cols = list(df2.columns) + [c for c in df1.columns if c != key]
        return pd.DataFrame(columns=final_cols)
        
    return pd.concat(partition_results, ignore_index=True)


# --------------------------------------------------------------------------
# 3. Hybrid Hash Join (In-Memory Partition 0 + Disk Buckets)
# --------------------------------------------------------------------------
def hybrid_hash_join(df1: pd.DataFrame, df2: pd.DataFrame, key: str, num_partitions: int = 8) -> pd.DataFrame:
    """
    Performs a Hybrid Hash Join keeping Partition 0 in-memory.
    """
    cols1 = df1.columns
    cols2 = df2.columns
    
    partition_0_hash: Dict[object, List[tuple]] = {}
    disk_buckets_1 = [[] for _ in range(1, num_partitions)]
    disk_buckets_2 = [[] for _ in range(1, num_partitions)]
    
    # --- Build Phase (df1) ---
    for row in df1.itertuples(index=False, name=None):
        row_dict = dict(zip(cols1, row))
        val = row_dict.get(key)
        if pd.isna(val):
            continue
        
        p_idx = abs(hash(val)) % num_partitions
        if p_idx == 0:
            if val not in partition_0_hash:
                partition_0_hash[val] = []
            partition_0_hash[val].append(row)
        else:
            disk_buckets_1[p_idx - 1].append(row)
            
    # --- Probe Phase & Partitioning for df2 ---
    joined_rows = []
    
    for row in df2.itertuples(index=False, name=None):
        row_dict = dict(zip(cols2, row))
        val = row_dict.get(key)
        if pd.isna(val):
            continue
            
        p_idx = abs(hash(val)) % num_partitions
        if p_idx == 0:
            if val in partition_0_hash:
                for b_row in partition_0_hash[val]:
                    b_dict = dict(zip(cols1, b_row))
                    del b_dict[key]
                    joined_rows.append({**row_dict, **b_dict})
        else:
            disk_buckets_2[p_idx - 1].append(row)
            
    # --- Join Remaining Partitions ---
    partition_results = [pd.DataFrame(joined_rows)] if joined_rows else []
    
    for i in range(num_partitions - 1):
        b1_df = pd.DataFrame(disk_buckets_1[i], columns=cols1) if disk_buckets_1[i] else pd.DataFrame(columns=cols1)
        b2_df = pd.DataFrame(disk_buckets_2[i], columns=cols2) if disk_buckets_2[i] else pd.DataFrame(columns=cols2)
        
        if not b1_df.empty and not b2_df.empty:
            res = simple_hash_join(b1_df, b2_df, key)
            if not res.empty:
                partition_results.append(res)
                
    if not partition_results:
        final_cols = list(cols2) + [c for c in cols1 if c != key]
        return pd.DataFrame(columns=final_cols)
        
    return pd.concat(partition_results, ignore_index=True)


# --------------------------------------------------------------------------
# 4. Radix Hash Join (Cache-conscious Vectorized Partitioning)
# --------------------------------------------------------------------------
def radix_hash_join(df1: pd.DataFrame, df2: pd.DataFrame, key: str, num_radix_bits: int = 4) -> pd.DataFrame:
    """
    Performs a Radix Hash Join. Uses true bitwise masking if keys are integers
    for optimal performance.
    """
    num_partitions = 2 ** num_radix_bits
    mask = num_partitions - 1
    
    # Check if keys are integers for true radix bitwise masking speedup
    is_int_key = (
        pd.api.types.is_integer_dtype(df1[key]) and 
        pd.api.types.is_integer_dtype(df2[key])
    )
    
    if is_int_key:
        h_keys1 = df1[key].apply(lambda x: (x & mask) if pd.notna(x) else -1)
        h_keys2 = df2[key].apply(lambda x: (x & mask) if pd.notna(x) else -1)
    else:
        h_keys1 = df1[key].apply(lambda x: (abs(hash(x)) & mask) if pd.notna(x) else -1)
        h_keys2 = df2[key].apply(lambda x: (abs(hash(x)) & mask) if pd.notna(x) else -1)
        
    valid_df1 = df1[h_keys1 != -1]
    valid_keys1 = h_keys1[h_keys1 != -1]
    
    valid_df2 = df2[h_keys2 != -1]
    valid_keys2 = h_keys2[h_keys2 != -1]
    
    groups1 = {p: grp for p, grp in valid_df1.groupby(valid_keys1)}
    groups2 = {p: grp for p, grp in valid_df2.groupby(valid_keys2)}
    
    partition_results = []
    common_partitions = set(groups1.keys()) & set(groups2.keys())
    
    for p_idx in common_partitions:
        res = simple_hash_join(groups1[p_idx], groups2[p_idx], key)
        if not res.empty:
            partition_results.append(res)
            
    if not partition_results:
        final_cols = list(df2.columns) + [c for c in df1.columns if c != key]
        return pd.DataFrame(columns=final_cols)
        
    return pd.concat(partition_results, ignore_index=True)