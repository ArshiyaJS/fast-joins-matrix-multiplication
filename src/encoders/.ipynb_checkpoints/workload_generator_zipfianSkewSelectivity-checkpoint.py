import pandas as pd
import numpy as np
from typing import Tuple, List

def generate_zipfian_relation(
    num_tuples: int,
    domain_size_A: int,
    domain_size_B: int,
    skew_factor_B: float,
    col_A_name: str = 'A',
    col_B_name: str = 'B',
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a 2-column DataFrame with a Zipfian distribution on the second column.
    """
    rng = np.random.default_rng(seed)
    domain_A = np.arange(domain_size_A)
    column_A = rng.choice(domain_A, size=num_tuples, replace=True)

    if skew_factor_B > 0:
        zipf_param = skew_factor_B + 1
        column_B = rng.zipf(a=zipf_param, size=num_tuples) - 1
        column_B = np.clip(column_B, 0, domain_size_B - 1)
    else:
        domain_B = np.arange(domain_size_B)
        column_B = rng.choice(domain_B, size=num_tuples, replace=True)

    df = pd.DataFrame({
        col_A_name: column_A,
        col_B_name: column_B
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


class MultiTopologyWorkloadGenerator:
    """
    A class to generate synthetic data for different join query topologies,
    with precise control over data skew and structure.
    """
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed

    def generate_matrix_query(
        self, 
        N: int, 
        n_A: int, 
        n_B: int, 
        n_C: int, 
        skew_b: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generates Q_matrix: R(A, B) JOIN S(B, C)"""
        R = generate_zipfian_relation(
            num_tuples=N, domain_size_A=n_A, domain_size_B=n_B, 
            skew_factor_B=skew_b, col_A_name='A', col_B_name='B', seed=self.base_seed
        )
        S = generate_zipfian_relation(
            num_tuples=N, domain_size_A=n_C, domain_size_B=n_B, 
            skew_factor_B=skew_b, col_A_name='C', col_B_name='B', seed=self.base_seed + 1
        )
        S = S[['B', 'C']]
        return R, S

    def generate_star_query(
        self, 
        N: int, 
        num_leaves: int, 
        domain_center_b: int, 
        skew_b: float
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """Generates Q_star: A central table joined with multiple leaf tables on key B."""
        center_table = generate_zipfian_relation(
            num_tuples=N, domain_size_A=int(N * 0.5), domain_size_B=domain_center_b,
            skew_factor_B=skew_b, col_A_name='id', col_B_name='B', seed=self.base_seed
        )
        leaves = []
        for i in range(num_leaves):
            leaf_df = generate_zipfian_relation(
                num_tuples=int(N * 0.8), domain_size_A=int(N * 0.4), domain_size_B=domain_center_b,
                skew_factor_B=skew_b, col_A_name=f'val_{i+1}', col_B_name='B', seed=self.base_seed + 10 + i
            )
            leaves.append(leaf_df[['B', f'val_{i+1}']])
        return center_table, leaves

    def generate_acyclic_path_query(
        self, 
        N: int, 
        path_length: int, 
        domain_sizes: List[int], 
        skew_factors: List[float]
    ) -> List[pd.DataFrame]:
        """Generates a line query: R1(X0,X1) ⨝ R2(X1,X2) ⨝ ..."""
        if len(domain_sizes) != path_length + 1 or len(skew_factors) != path_length:
            raise ValueError("Domain/skew list sizes do not match path length.")
            
        relations = []
        for i in range(path_length):
            df = generate_zipfian_relation(
                num_tuples=N,
                domain_size_A=domain_sizes[i],
                domain_size_B=domain_sizes[i+1],
                skew_factor_B=skew_factors[i],
                col_A_name=f'X_{i}',
                col_B_name=f'X_{i+1}',
                seed=self.base_seed + i * 100
            )
            relations.append(df)
        return relations
