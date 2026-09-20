import pandas as pd
from typing import List, Dict, Tuple, Any

# ==========================================================================
# 1. Simple Hash Join (Upgraded with Suffixes and Diagnostics)
# ==========================================================================
def simple_hash_join(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    key: str, 
    lsuffix: str = '_left', 
    rsuffix: str = '_right',
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Performs a classic in-memory hash join with column collision handling.
    """
    if len(df1) <= len(df2):
        build_df, probe_df = df1, df2
        build_suffix, probe_suffix = lsuffix, rsuffix
    else:
        build_df, probe_df = df2, df1
        build_suffix, probe_suffix = rsuffix, lsuffix

    # --- Handle Column Collisions ---
    build_cols = list(build_df.columns)
    probe_cols = list(probe_df.columns)
    colliding_cols = [c for c in build_cols if c in probe_cols and c != key]
    
    renamed_build_cols = {c: c + build_suffix for c in colliding_cols}
    build_df = build_df.rename(columns=renamed_build_cols)
    build_cols = list(build_df.columns)

    # --- Build Phase ---
    hash_table: Dict[Any, List[tuple]] = {}
    for row in build_df.itertuples(index=False, name=None):
        row_dict = dict(zip(build_cols, row))
        key_val = row_dict.get(key)
        if pd.notna(key_val):
            hash_table.setdefault(key_val, []).append(row)

    # --- Probe Phase ---
    joined_rows = []
    for probe_row_tuple in probe_df.itertuples(index=False, name=None):
        probe_row_dict = dict(zip(probe_cols, probe_row_tuple))
        key_val = probe_row_dict.get(key)
        
        if key_val in hash_table:
            for build_row_tuple in hash_table[key_val]:
                build_row_dict = dict(zip(build_cols, build_row_tuple))
                del build_row_dict[key]
                joined_rows.append({**probe_row_dict, **build_row_dict})

    diagnostics = {'build_table_size': len(build_df), 'probe_table_size': len(probe_df)}

    if not joined_rows:
        final_cols = probe_cols + [c for c in build_cols if c != key]
        return pd.DataFrame(columns=final_cols), diagnostics
        
    return pd.DataFrame(joined_rows), diagnostics


# ==========================================================================
# 2. Grace Hash Join (Upgraded with Diagnostics and Vectorization)
# ==========================================================================
def grace_hash_join(df1: pd.DataFrame, df2: pd.DataFrame, key: str, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Performs a partition-based Grace Hash Join with skew telemetry.
    """
    num_partitions = kwargs.get('num_partitions', 8)
    
    h_keys1 = df1[key].apply(lambda x: abs(hash(x)) % num_partitions if pd.notna(x) else -1)
    h_keys2 = df2[key].apply(lambda x: abs(hash(x)) % num_partitions if pd.notna(x) else -1)
    
    groups1 = {p: grp for p, grp in df1[h_keys1 != -1].groupby(h_keys1[h_keys1 != -1])}
    groups2 = {p: grp for p, grp in df2[h_keys2 != -1].groupby(h_keys2[h_keys2 != -1])}
    
    partition_skew_diag = {
        "df1_partition_sizes": {p: len(grp) for p, grp in groups1.items()},
        "df2_partition_sizes": {p: len(grp) for p, grp in groups2.items()}
    }

    partition_results = []
    common_partitions = set(groups1.keys()) & set(groups2.keys())
    
    for p_idx in common_partitions:
        res, _ = simple_hash_join(groups1[p_idx], groups2[p_idx], key, **kwargs)
        if not res.empty:
            partition_results.append(res)
            
    if not partition_results:
        lsuffix = kwargs.get('lsuffix', '_left')
        final_cols = list(df2.columns) + [c if c not in df2.columns or c == key else c + lsuffix for c in df1.columns]
        return pd.DataFrame(columns=final_cols), partition_skew_diag
        
    return pd.concat(partition_results, ignore_index=True), partition_skew_diag


# ==========================================================================
# 3. Hybrid Hash Join (Partition 0 In-Memory + Disk Buckets)
# ==========================================================================
def hybrid_hash_join(df1: pd.DataFrame, df2: pd.DataFrame, key: str, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Performs a Hybrid Hash Join capturing Partition 0 in RAM while bucketing the rest.
    """
    num_partitions = kwargs.get('num_partitions', 8)
    lsuffix = kwargs.get('lsuffix', '_left')
    rsuffix = kwargs.get('rsuffix', '_right')
    
    cols1 = list(df1.columns)
    cols2 = list(df2.columns)
    colliding_cols = [c for c in cols1 if c in cols2 and c != key]
    renamed_cols1 = {c: c + lsuffix for c in colliding_cols}
    df1_renamed = df1.rename(columns=renamed_cols1)
    cols1 = list(df1_renamed.columns)

    partition_0_hash: Dict[Any, List[tuple]] = {}
    disk_buckets_1 = [[] for _ in range(1, num_partitions)]
    disk_buckets_2 = [[] for _ in range(1, num_partitions)]
    
    p0_df1_count = 0
    p0_df2_count = 0
    disk_counts_1 = {i: 0 for i in range(1, num_partitions)}
    disk_counts_2 = {i: 0 for i in range(1, num_partitions)}

    # --- Build Phase (df1) ---
    for row in df1_renamed.itertuples(index=False, name=None):
        row_dict = dict(zip(cols1, row))
        val = row_dict.get(key)
        if pd.isna(val):
            continue
        p_idx = abs(hash(val)) % num_partitions
        if p_idx == 0:
            p0_df1_count += 1
            partition_0_hash.setdefault(val, []).append(row)
        else:
            disk_counts_1[p_idx] += 1
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
            p0_df2_count += 1
            if val in partition_0_hash:
                for b_row in partition_0_hash[val]:
                    b_dict = dict(zip(cols1, b_row))
                    del b_dict[key]
                    joined_rows.append({**row_dict, **b_dict})
        else:
            disk_counts_2[p_idx] += 1
            disk_buckets_2[p_idx - 1].append(row)
            
    diagnostics = {
        "df1_partition_sizes": {0: p0_df1_count, **disk_counts_1},
        "df2_partition_sizes": {0: p0_df2_count, **disk_counts_2}
    }

    partition_results = [pd.DataFrame(joined_rows)] if joined_rows else []
    
    for i in range(num_partitions - 1):
        b1_df = pd.DataFrame(disk_buckets_1[i], columns=cols1) if disk_buckets_1[i] else pd.DataFrame(columns=cols1)
        b2_df = pd.DataFrame(disk_buckets_2[i], columns=cols2) if disk_buckets_2[i] else pd.DataFrame(columns=cols2)
        
        if not b1_df.empty and not b2_df.empty:
            res, _ = simple_hash_join(b1_df, b2_df, key, lsuffix=lsuffix, rsuffix=rsuffix)
            if not res.empty:
                partition_results.append(res)
                
    if not partition_results:
        final_cols = cols2 + [c for c in cols1 if c != key]
        return pd.DataFrame(columns=final_cols), diagnostics
        
    return pd.concat(partition_results, ignore_index=True), diagnostics


# ==========================================================================
# 4. Radix Hash Join (Cache-conscious Partitioning)
# ==========================================================================
def radix_hash_join(df1: pd.DataFrame, df2: pd.DataFrame, key: str, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Performs a Radix Hash Join using bitwise masks for cache locality.
    """
    num_radix_bits = kwargs.get('num_radix_bits', 4)
    num_partitions = 2 ** num_radix_bits
    mask = num_partitions - 1
    
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
    
    diagnostics = {
        "df1_partition_sizes": {p: len(grp) for p, grp in groups1.items()},
        "df2_partition_sizes": {p: len(grp) for p, grp in groups2.items()}
    }
    
    partition_results = []
    common_partitions = set(groups1.keys()) & set(groups2.keys())
    
    for p_idx in common_partitions:
        res, _ = simple_hash_join(groups1[p_idx], groups2[p_idx], key, **kwargs)
        if not res.empty:
            partition_results.append(res)
            
    if not partition_results:
        lsuffix = kwargs.get('lsuffix', '_left')
        final_cols = list(df2.columns) + [c if c not in df2.columns or c == key else c + lsuffix for c in df1.columns]
        return pd.DataFrame(columns=final_cols), diagnostics
        
    return pd.concat(partition_results, ignore_index=True), diagnostics