import pandas as pd
from typing import Tuple, Dict

def sort_merge_join(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    key: str, 
    lsuffix: str = '_left', 
    rsuffix: str = '_right',
    **kwargs
) -> Tuple[pd.DataFrame, Dict]:
    """
    Performs a classic Sort-Merge join with robust handling of duplicates and NULLs.
    """
    # --- Sort Phase ---
    # Sort both DataFrames on the join key. Place NULLs at the beginning.
    df1_sorted = df1.sort_values(by=key, ascending=True, na_position='first').reset_index(drop=True)
    df2_sorted = df2.sort_values(by=key, ascending=True, na_position='first').reset_index(drop=True)

    # --- Handle Column Collisions ---
    cols1 = list(df1_sorted.columns)
    cols2 = list(df2_sorted.columns)
    colliding_cols = [c for c in cols1 if c in cols2 and c != key]
    
    renamed_cols1 = {c: c + lsuffix for c in colliding_cols}
    df1_sorted = df1_sorted.rename(columns=renamed_cols1)
    cols1 = list(df1_sorted.columns)

    # --- Merge Phase ---
    ptr1, ptr2 = 0, 0
    len1, len2 = len(df1_sorted), len(df2_sorted)
    joined_rows = []

    while ptr1 < len1 and ptr2 < len2:
        # Skip any NULL keys at the current pointer positions
        while ptr1 < len1 and pd.isna(df1_sorted.loc[ptr1, key]):
            ptr1 += 1
        while ptr2 < len2 and pd.isna(df2_sorted.loc[ptr2, key]):
            ptr2 += 1
        
        # If pointers have reached the end after skipping NULLs, break
        if ptr1 >= len1 or ptr2 >= len2:
            break

        key1 = df1_sorted.loc[ptr1, key]
        key2 = df2_sorted.loc[ptr2, key]

        if key1 < key2:
            ptr1 += 1
        elif key1 > key2:
            ptr2 += 1
        else:  # Keys match, handle block cartesian product
            # Find the end of the duplicate block in df1
            block_end1 = ptr1
            while block_end1 < len1 and df1_sorted.loc[block_end1, key] == key1:
                block_end1 += 1
            
            # Find the end of the duplicate block in df2
            block_end2 = ptr2
            while block_end2 < len2 and df2_sorted.loc[block_end2, key] == key2:
                block_end2 += 1
                
            # Perform the Cartesian product for the blocks
            for i in range(ptr1, block_end1):
                row1_dict = df1_sorted.iloc[i].to_dict()
                for j in range(ptr2, block_end2):
                    row2_dict = df2_sorted.iloc[j].to_dict()
                    temp_row2 = row2_dict.copy()
                    del temp_row2[key]
                    joined_rows.append({**row1_dict, **temp_row2})
            
            # Move pointers to the start of the next distinct block
            ptr1 = block_end1
            ptr2 = block_end2

    diagnostics = {"info": "Sort-Merge Join Executed"}
    
    if not joined_rows:
        final_cols = cols1 + [c for c in cols2 if c != key]
        return pd.DataFrame(columns=final_cols), diagnostics

    return pd.DataFrame(joined_rows), diagnostics
