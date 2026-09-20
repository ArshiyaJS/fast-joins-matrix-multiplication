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

        # Edge Case: Ignore NULLs in the join key
        if pd.isna(key_val):
            continue
        
        # If the key is not in the table, create a new list for it
        if key_val not in hash_table:
            hash_table[key_val] = []
        
        # Edge Case: Handle duplicate keys by appending to the list
        hash_table[key_val].append(row)

    # --- Probe Phase ---
    joined_rows = []
    probe_cols = probe_df.columns
    
    for probe_row_tuple in probe_df.itertuples(index=False, name=None):
        probe_row_dict = dict(zip(probe_cols, probe_row_tuple))
        key_val = probe_row_dict.get(key)
        
        # Find a match in the hash table
        if key_val in hash_table:
            # Join with every matching row from the build table
            for build_row_tuple in hash_table[key_val]:
                build_row_dict = dict(zip(build_cols, build_row_tuple))
                
                # Combine the rows, excluding the duplicate join key from one side
                del build_row_dict[key]
                joined_row = {**probe_row_dict, **build_row_dict}
                joined_rows.append(joined_row)

    if not joined_rows:
        # Construct an empty DataFrame with the correct columns if no join occurred
        final_cols = list(probe_cols) + [c for c in build_cols if c != key]
        return pd.DataFrame(columns=final_cols)
        
    return pd.DataFrame(joined_rows)

# --------------------------------------------------------------------------
# 2. Grace Hash Join (Partition-based for large datasets)
# --------------------------------------------------------------------------
def grace_hash_join(df1: pd.DataFrame, df2: pd.DataFrame, key: str, num_partitions: int = 8) -> pd.DataFrame:
    """
    Performs a Grace Hash Join. This is a two-pass algorithm.
    Pass 1: Partition both tables using a hash function.
    Pass 2: Perform a simple_hash_join on each pair of partitions.
    """
    
    # --- Partitioning Phase ---
    # 1. Create `num_partitions` empty buckets (lists of DataFrames) for each table.
    buckets_1 = [[] for _ in range(num_partitions)]
    buckets_2 = [[] for _ in range(num_partitions)]
    
    cols1 = df1.columns
    cols2 = df2.columns
    # 2. Iterate through df1: hash the key, and append the row to the correct bucket.
    for row in df1.itertuples(index=False, name=None):
        row_dict = dict(zip(cols1, row))
        val = row_dict.get(key)
        if pd.isna(val):
            continue
        p_idx = abs(hash(val)) % num_partitions
        buckets_1[p_idx].append(row)
    # 3. Iterate through df2: hash the key, and append the row to the correct bucket.
    for row in df2.itertuples(index=False, name=None):
        row_dict = dict(zip(cols2, row))
        val = row_dict.get(key)
        if pd.isna(val):
            continue
        p_idx = abs(hash(val)) % num_partitions
        buckets_2[p_idx].append(row)
    
    # --- Join Phase ---
    # 4. Initialize an empty list for final results.
    partition_results = []
    # 5. Loop from i = 0 to num_partitions-1:
    for i in range(num_partitions):
        # - Call `simple_hash_join` on the partition pair (bucket_R[i], bucket_S[i]).
        b1_df = pd.DataFrame(buckets_1[i], columns=cols1) if buckets_1[i] else pd.DataFrame(columns=cols1)
        b2_df = pd.DataFrame(buckets_2[i], columns=cols2) if buckets_2[i] else pd.DataFrame(columns=cols2)

        if not b1_df.empty and not b2_df.empty:
            res = simple_hash_join(b1_df, b2_df, key)
            if not res.empty:
                partition_results.append(res)
    
    if not partition_results:
        final_cols = list(df2.columns) + [c for c in df1.columns if c != key]
    # 6. Concatenate all results into a single DataFrame.
        return pd.DataFrame(columns=final_cols)
    
    return pd.concat(partition_results, ignore_index=True)

# --------------------------------------------------------------------------
# 3. Hybrid Hash Join (Optimization of Grace Hash Join)
# --------------------------------------------------------------------------
def hybrid_hash_join(df1: pd.DataFrame, df2: pd.DataFrame, key: str, num_partitions: int = 8) -> pd.DataFrame:
    """
    Performs a Hybrid Hash Join.
    It's similar to Grace, but it keeps the first partition in memory
    to avoid writing it to/reading it from disk, saving I/O costs.
    """
    # --- Partitioning Phase (with optimization) ---
    cols1 = df1.columns
    cols2 = df2.columns
    # 1. The first partition (bucket 0) is kept in-memory as a hash table.
    partition_0_hash: Dict[object, List[tuple]] = {}
    # 2. The other k-1 partitions are written to disk (or kept as in-memory DataFrames for this simulation).
    disk_buckets_1 = [[] for _ in range(1, num_partitions)]
    disk_buckets_2 = [[] for _ in range(1, num_partitions)]
    # 3. As you partition the second table, if a row belongs to partition 0, join it immediately.
    #    If it belongs to other partitions, write it to the corresponding disk bucket.

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
            # Probe Partition 0 immediately on the fly
            if val in partition_0_hash:
                for b_row in partition_0_hash[val]:
                    b_dict = dict(zip(cols1, b_row))
                    del b_dict[key]
                    joined_rows.append({**row_dict, **b_dict})
        else:
            disk_buckets_2[p_idx - 1].append(row)

    # --- Join Phase ---
    # 4. Join the remaining k-1 partitions on disk just like in Grace join.
    partition_results = [pd.DataFrame(joined_rows)] if joined_rows else []
    
    for i in range(num_partitions - 1):
        b1_df = pd.DataFrame(disk_buckets_1[i], columns=cols1) if disk_buckets_1[i] else pd.DataFrame(columns=cols1)
        b2_df = pd.DataFrame(disk_buckets_2[i], columns=cols2) if disk_buckets_2[i] else pd.DataFrame(columns=cols2)
        
        if not b1_df.empty and not b2_df.empty:
            res = simple_hash_join(b1_df, b2_df, key)
            if not res.empty:
                partition_results.append(res)
    # 5. Combine the results.
    if not partition_results:
        final_cols = list(cols2) + [c for c in cols1 if c != key]
        return pd.DataFrame(columns=final_cols)
        
    return pd.concat(partition_results, ignore_index=True)

# --------------------------------------------------------------------------
# 4. Radix Hash Join (Cache-conscious Partitioning)
# --------------------------------------------------------------------------
def radix_hash_join(df1: pd.DataFrame, df2: pd.DataFrame, key: str, num_radix_bits: int = 4) -> pd.DataFrame:
    num_partitions = 2 ** num_radix_bits
    mask = num_partitions - 1
    
    buckets_1 = [[] for _ in range(num_partitions)]
    buckets_2 = [[] for _ in range(num_partitions)]
    
    cols1 = df1.columns
    cols2 = df2.columns
    
    # Radix partitioning using bitwise AND mask
    for row in df1.itertuples(index=False, name=None):
        row_dict = dict(zip(cols1, row))
        val = row_dict.get(key)
        if pd.isna(val):
            continue
        p_idx = abs(hash(val)) & mask
        buckets_1[p_idx].append(row)
        
    for row in df2.itertuples(index=False, name=None):
        row_dict = dict(zip(cols2, row))
        val = row_dict.get(key)
        if pd.isna(val):
            continue
        p_idx = abs(hash(val)) & mask
        buckets_2[p_idx].append(row)
        
    # Join micro-partitions to leverage cache residency
    partition_results = []
    for i in range(num_partitions):
        b1_df = pd.DataFrame(buckets_1[i], columns=cols1) if buckets_1[i] else pd.DataFrame(columns=cols1)
        b2_df = pd.DataFrame(buckets_2[i], columns=cols2) if buckets_2[i] else pd.DataFrame(columns=cols2)
        
        if not b1_df.empty and not b2_df.empty:
            res = simple_hash_join(b1_df, b2_df, key)
            if not res.empty:
                partition_results.append(res)
                
    if not partition_results:
        final_cols = list(cols2) + [c for c in cols1 if c != key]
        return pd.DataFrame(columns=final_cols)
        
    return pd.concat(partition_results, ignore_index=True)
