import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any
from itertools import product

def seek(sorted_array: np.ndarray, value: Any) -> Tuple[Any, int]:
    """
    Core Leapfrog primitive: Finds the first element in sorted_array that is >= value.
    Leverages np.searchsorted for O(log N) binary search performance.
    """
    idx = np.searchsorted(sorted_array, value, side='left')
    if idx < len(sorted_array):
        return sorted_array[idx], idx
    return None, -1


def lftj_binary_join(df1: pd.DataFrame, df2: pd.DataFrame, key: str, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    A Leapfrog-style two-table join (similar to sort-merge join mechanics).
    """
    lsuffix = kwargs.get('lsuffix', '_left')
    rsuffix = kwargs.get('rsuffix', '_right')

    # Handle colliding non-key columns safely
    cols1, cols2 = list(df1.columns), list(df2.columns)
    colliding = [c for c in cols1 if c in cols2 and c != key]
    df1 = df1.rename(columns={c: c + lsuffix for c in colliding})
    df2 = df2.rename(columns={c: c + rsuffix for c in colliding})

    # Build Tries (Index dictionaries mapping key -> list of row dicts)
    index1 = {k: g.to_dict('records') for k, g in df1.groupby(key)}
    index2 = {k: g.to_dict('records') for k, g in df2.groupby(key)}

    # Find common keys via set intersection (The initial leap/alignment phase)
    common_keys = sorted(list(set(index1.keys()) & set(index2.keys())))

    joined_rows = []
    for k in common_keys:
        rows1 = index1[k]
        rows2 = index2[k]
        for r1 in rows1:
            for r2 in rows2:
                r2_copy = r2.copy()
                del r2_copy[key]
                joined_rows.append({**r1, **r2_copy})

    diagnostics = {"info": "LFTJ Binary Join Executed", "output_rows": len(joined_rows)}
    
    if not joined_rows:
        final_cols = list(df1.columns) + [c for c in df2.columns if c != key]
        return pd.DataFrame(columns=final_cols), diagnostics

    return pd.DataFrame(joined_rows), diagnostics


def lftj_triangle_join(R_AB: pd.DataFrame, S_BC: pd.DataFrame, T_CA: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Specialized LFTJ implementation for the Triangle Query (R(A,B) ⨝ S(B,C) ⨝ T(C,A)).
    Uses explicit variable ordering: A -> B -> C with binary search leaping.
    """
    # --- Indexing Phase: Convert columns to sorted NumPy arrays for O(log N) lookups ---
    r_index = {key: group['B'].sort_values().values for key, group in R_AB.groupby('A')}
    s_index = {key: group['C'].sort_values().values for key, group in S_BC.groupby('B')}
    t_index = {key: group['A'].sort_values().values for key, group in T_CA.groupby('C')}

    triangles = []
    
    # --- Join Phase (Variable Order: A -> B -> C) ---
    sorted_a_keys = sorted(r_index.keys())

    for a_val in sorted_a_keys:
        b_candidates = r_index[a_val]
        
        for b_val in b_candidates:
            # Leap 1: Check if b_val exists in S's index
            if b_val in s_index:
                c_candidates = s_index[b_val]
                
                for c_val in c_candidates:
                    # Leap 2: Check if c_val exists in T's index
                    if c_val in t_index:
                        sorted_a_for_c = t_index[c_val]
                        
                        # Leap 3 (Final Check): Binary search in T to close the triangle (c -> a)
                        val_found, idx = seek(sorted_a_for_c, a_val)
                        
                        if val_found == a_val:
                            triangles.append({'A': a_val, 'B': b_val, 'C': c_val})
                            
    diagnostics = {"info": "LFTJ Triangle Join Executed", "output_rows": len(triangles)}
    return pd.DataFrame(triangles), diagnostics


def lftj_star_join(center_df: pd.DataFrame, leaf_dfs: List[pd.DataFrame], key: str, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Specialized LFTJ for a Star Query centered around a primary join key.
    """
    all_indices = []
    
    # Index center table
    all_indices.append({k: g.to_dict('records') for k, g in center_df.groupby(key)})

    # Index all leaf tables with namespace conflict prevention
    for i, leaf in enumerate(leaf_dfs):
        cols_to_rename = [c for c in leaf.columns if c in center_df.columns and c != key]
        renamed_leaf = leaf.rename(columns={c: f"{c}_leaf{i}" for c in cols_to_rename})
        all_indices.append({k: g.to_dict('records') for k, g in renamed_leaf.groupby(key)})

    if not all_indices:
        return pd.DataFrame(), {"info": "No tables provided."}

    # Leap 1: Find the intersection of keys across ALL tables simultaneously
    common_keys = set(all_indices[0].keys())
    for idx_dict in all_indices[1:]:
        common_keys.intersection_update(idx_dict.keys())

    joined_rows = []
    for k in sorted(list(common_keys)):
        row_groups = [idx_dict[k] for idx_dict in all_indices]
        
        # Multi-way Cartesian product for matching records under key `k`
        for combination in product(*row_groups):
            merged_row = {}
            for row_dict in combination:
                merged_row.update(row_dict)
            
            # Clean up duplicate keys
            if key in merged_row:
                del merged_row[key]
            merged_row[key] = k
            
            joined_rows.append(merged_row)

    diagnostics = {"info": "LFTJ Star Join Executed", "output_rows": len(joined_rows)}
    return pd.DataFrame(joined_rows), diagnostics