import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

class SharedDomainEncoder:
    def __init__(self):
        self.row_mapping = None
        self.join_mapping = None
        self.col_mapping = None

    def fit_and_encode(self, df1, df2, left_on, right_on, left_row, right_col):
        """
        Encodes df1 and df2, creating both sparse and dense matrix representations
        with perfectly aligned join keys.
        """
        # 1. Create the global, shared mappings for all unique values
        self.row_mapping = pd.Index(df1[left_row].unique())
        self.join_mapping = pd.Index(pd.concat([df1[left_on], df2[right_on]]).unique())
        self.col_mapping = pd.Index(df2[right_col].unique())

        # --- Helper function to get matrix components ---
        def get_matrix_components(df, row_name, col_name, row_map, col_map):
            row_idx = row_map.get_indexer(df[row_name])
            col_idx = col_map.get_indexer(df[col_name])
            
            # Use a mask to filter out any values that might not be in the map
            valid = (row_idx >= 0) & (col_idx >= 0)
            
            data = np.ones(np.sum(valid), dtype=int)
            shape = (len(row_map), len(col_map))
            
            return data, row_idx[valid], col_idx[valid], shape

        # --- Encode Relation 1 (R) ---
        data_r, rows_r, cols_r, shape_r = get_matrix_components(df1, left_row, left_on, self.row_mapping, self.join_mapping)
        # Sparse version of R
        sparse_R = csr_matrix((data_r, (rows_r, cols_r)), shape=shape_r)
        # Dense version of R
        dense_R = sparse_R.toarray()
        
        # --- Encode Relation 2 (S) ---
        data_s, rows_s, cols_s, shape_s = get_matrix_components(df2, right_on, right_col, self.join_mapping, self.col_mapping)
        # Sparse version of S
        sparse_S = csr_matrix((data_s, (rows_s, cols_s)), shape=shape_s)
        # Dense version of S
        dense_S = sparse_S.toarray()

        return (sparse_R, dense_R), (sparse_S, dense_S)

    def decode_result(self, result_matrix):
        """
        Converts a resulting matrix (can be sparse or dense) back into a 
        relational Pandas DataFrame.
        """
        # Ensure matrix is in COO format for easy coordinate access
        if not isinstance(result_matrix, coo_matrix):
            result_matrix = coo_matrix(result_matrix)
        
        decoded_df = pd.DataFrame({
            'Left_Key': self.row_mapping[result_matrix.row],
            'Right_Key': self.col_mapping[result_matrix.col],
            'Match_Weight': result_matrix.data
        })
        return decoded_df

