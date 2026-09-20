import time
from src.encoders.workload_generator_zipfianSkewSelectivity import MultiTopologyWorkloadGenerator

# from src.joins import execute_q_matrix_hash, execute_q_matrix_spgemm

def main():
    print("--- Initializing Workload Generator ---")
    generator = MultiTopologyWorkloadGenerator(base_seed=42)

    # --- 1. Generate and test Q_matrix ---
    print("\n--- Generating data for Q_matrix ---")
    R, S = generator.generate_matrix_query(N=100_000, n_A=50_000, n_B=1_000, n_C=50_000, skew_b=1.2)
    print("Generated R(A, B) and S(B, C)")
    print(f"R length: {len(R)}, S length: {len(S)}")
    # join functions:
    # time_hash = time_function(execute_q_matrix_hash, R, S)
    # time_spgemm = time_function(execute_q_matrix_spgemm, R, S)

    print("\n--- R(A, B) Head ---")
    print(R.head())
    
    print("\n--- S(B, C) Head ---")
    print(S.head())
    
    # --- 2. Generate and test Q_star ---
    print("\n--- Generating data for Q_star ---")
    center, leaves = generator.generate_star_query(N=100_000, num_leaves=3, domain_center_b=1_000, skew_b=1.5)
    print(f"Generated center table and {len(leaves)} leaf tables")
    # star join functions

    print("\n--- Center Table(id, B) Head ---")
    print(center.head())

    for i, leaf_df in enumerate(leaves):
        print(f"\n--- Leaf Table {i+1} Head (B, val_{i+1}) ---")
        print(leaf_df.head())

    # --- 3. Generate and test Q_line ---
    print("\n--- Generating data for Q_line ---")
    # 3-relation path: R1(X0, X1) -> R2(X1, X2) -> R3(X2, X3)
    path_relations = generator.generate_acyclic_path_query(
        N=100_000, 
        path_length=3, 
        domain_sizes=[10_000, 1_000, 1_000, 10_000], 
        skew_factors=[1.0, 1.2, 0.8]
    )
    print(f"Generated {len(path_relations)} relations for a line query")
    # line join functions

    for i, path_df in enumerate(path_relations):
        print(f"\n--- Path Relation {i+1} Head ({path_df.columns[0]}, {path_df.columns[1]}) ---")
        print(path_df.head())

if __name__ == "__main__":
    main()
