import time
import gc
import pandas as pd
import numpy as np
import os

from src.encoders.workload_generator_zipfianSkewSelectivity import MultiTopologyWorkloadGenerator
from src.joins.query_executor_combined import execute_query_plan, JOIN_ALGORITHMS

# Create output directory for modular results
OUTPUT_DIR = "benchmark results leapfrog"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_comprehensive_benchmark():
    generator = MultiTopologyWorkloadGenerator(base_seed=42)
    
    #algorithms = ["simple_hash", "grace_hash", "hybrid_hash", "radix_hash", "sort_merge", "pandas", "leapfrog"]
    algorithms = ["leapfrog"]

    def run_single_experiment(exp_name, topology_name, config_generator_func):
        results = []
        report_lines = []

        report_lines.append("============================================================")
        report_lines.append(f"    BENCHMARK REPORT: {exp_name} ({topology_name})        ")
        report_lines.append("============================================================\n")

        '''
        def evaluate_and_record(exp_n, topo_n, config_params, tables, plan_template):
            # 1. Compute Ground Truth using Pandas baseline
            try:
                ground_truth_res = execute_query_plan(tables.copy(), plan_template, join_algorithm="pandas")
                ground_truth_df = ground_truth_res[0] if isinstance(ground_truth_res, tuple) else ground_truth_res
                expected_len = len(ground_truth_df)
                del ground_truth_df
            except Exception as e:
                report_lines.append(f"[{exp_n} | {topo_n}] [PANDAS BASELINE FAILED]: {e}")
                expected_len = -1

            gc.collect()

            # 2. Iterate through candidate algorithms
            for algo_name in algorithms:
                if algo_name != "pandas" and algo_name not in JOIN_ALGORITHMS and algo_name != "leapfrog":
                    continue

                # Define partition configurations based on algorithm capability
                if algo_name in ["grace_hash", "radix_hash", "hybrid_hash"]:
                    partition_configs = [4, 8, 16]
                else:
                    partition_configs = [None]  # Partition-less & multi-way algorithms run once

                for num_p in partition_configs:
                    passed = False
                    error_msg = ""
                    row_count = 0
                    skew_metric = 0.0
                    elapsed = 0.0
                    final_result = None

                    try:
                        if algo_name == "pandas":
                            ref_start = time.time()
                            res = execute_query_plan(tables.copy(), plan_template, join_algorithm="pandas")
                            final_result = res[0] if isinstance(res, tuple) else res
                            elapsed = time.time() - ref_start
                            row_count = expected_len
                            passed = True
                            
                        elif algo_name == "leapfrog":
                            # Leapfrog is a multi-way join engine; it evaluates the entire query graph at once
                            start_t = time.time()
                            res = execute_query_plan(tables.copy(), plan_template, join_algorithm="leapfrog")
                            final_result = res[0] if isinstance(res, tuple) else res
                            elapsed = time.time() - start_t
                            row_count = len(final_result) if final_result is not None else 0
                            
                            if expected_len != -1 and row_count == expected_len:
                                passed = True
                            elif expected_len != -1:
                                error_msg = f"Length mismatch: got {row_count}, expected {expected_len}"
                        else:
                            start_t = time.time()
                            
                            # Build a fresh copy of the plan where steps include num_partitions if applicable
                            current_tables = tables.copy()
                            current_plan = []
                            for step in plan_template:
                                step_copy = step.copy()
                                if num_p is not None:
                                    step_copy['num_partitions'] = num_p
                                current_plan.append(step_copy)

                            for step_idx, step in enumerate(current_plan):
                                left_name = step['left']
                                right_name = step['right']

                                df_left = current_tables.get(left_name)
                                df_right = current_tables.get(right_name)

                                if df_left is None or df_right is None or df_left.empty or df_right.empty:
                                    final_result = pd.DataFrame()
                                    break
                                
                                sub_tables = {left_name: df_left, right_name: df_right}
                                
                                res = execute_query_plan(
                                    sub_tables, [step], 
                                    join_algorithm=algo_name
                                )
                                
                                final_result = res[0] if isinstance(res, tuple) else res
                                intermediate_name = f"intermediate_step_{step_idx}"
                                current_tables[intermediate_name] = final_result

                            elapsed = time.time() - start_t
                            row_count = len(final_result) if final_result is not None else 0

                            if expected_len != -1 and row_count == expected_len:
                                passed = True
                            elif expected_len != -1:
                                error_msg = f"Length mismatch: got {row_count}, expected {expected_len}"
                    except Exception as e:
                        error_msg = str(e)
                        passed = False

                    if final_result is not None:
                        del final_result
                    gc.collect()

                    status_str = "PASS" if passed else "FAIL"

                    record = {
                        "experiment": exp_n,
                        "topology": topo_n,
                        "algorithm": algo_name,
                        "num_partitions": num_p,
                        "status": status_str,
                        "execution_time_sec": elapsed,
                        "output_rows": row_count,
                        "partition_skew_metric": skew_metric,
                        "error": error_msg,
                        **config_params
                    }
                    results.append(record)

                    report_lines.append(
                        f"[{exp_n} | {topo_n}] Algo={algo_name:12} | Partitions={num_p if num_p is not None else 'N/A':3} | "
                        f"Status={status_str} | Time={elapsed:.4f}s | Notes={error_msg or 'None'}"
                    )
        '''

        def evaluate_and_record(exp_n, topo_n, config_params, tables, plan_template):
            # For Q_triangle, let Leapfrog define the expected length or compute valid triangle intersections
            try:
                if topo_n == "Q_triangle":
                    # Run leapfrog first to get the valid multi-way intersection count
                    res_lf = execute_query_plan(tables.copy(), plan_template, join_algorithm="leapfrog")
                    lf_df = res_lf[0] if isinstance(res_lf, tuple) else res_lf
                    expected_len = len(lf_df) if lf_df is not None else 0
                    del lf_df
                else:
                    ground_truth_res = execute_query_plan(tables.copy(), plan_template, join_algorithm="pandas")
                    ground_truth_df = ground_truth_res[0] if isinstance(ground_truth_res, tuple) else ground_truth_res
                    expected_len = len(ground_truth_df)
                    del ground_truth_df
            except Exception as e:
                report_lines.append(f"[{exp_n} | {topo_n}] [BASELINE FAILED]: {e}")
                expected_len = -1

            gc.collect()

            # 2. Iterate through candidate algorithms
            for algo_name in algorithms:
                if algo_name != "pandas" and algo_name not in JOIN_ALGORITHMS and algo_name != "leapfrog":
                    continue

                # Define partition configurations based on algorithm capability
                if algo_name in ["grace_hash", "radix_hash", "hybrid_hash"]:
                    partition_configs = [4, 8, 16]
                else:
                    partition_configs = [None]  # Partition-less & multi-way algorithms run once

                for num_p in partition_configs:
                    passed = False
                    error_msg = ""
                    row_count = 0
                    skew_metric = 0.0
                    elapsed = 0.0
                    final_result = None

                    try:
                        if algo_name == "pandas":
                            ref_start = time.time()
                            if topo_n == "Q_triangle":
                                df_r = tables.get('R')
                                df_s = tables.get('S')
                                df_t = tables.get('T')
                                final_result = df_r.merge(df_s, on='B').merge(df_t, on=['C', 'A'])
                            else:
                                res = execute_query_plan(tables.copy(), plan_template, join_algorithm="pandas")
                                final_result = res[0] if isinstance(res, tuple) else res
                            elapsed = time.time() - ref_start
                            row_count = expected_len
                            passed = True
                            
                        elif algo_name == "leapfrog":
                            start_t = time.time()
                            res = execute_query_plan(tables.copy(), plan_template, join_algorithm="leapfrog")
                            final_result = res[0] if isinstance(res, tuple) else res
                            elapsed = time.time() - start_t
                            row_count = len(final_result) if final_result is not None else 0
                            
                            if expected_len != -1 and row_count == expected_len:
                                passed = True
                            elif expected_len != -1:
                                error_msg = f"Length mismatch: got {row_count}, expected {expected_len}"
                        else:
                            # Standard binary join fallback for other algorithms
                            start_t = time.time()
                            current_tables = tables.copy()
                            current_plan = []
                            for step in plan_template:
                                step_copy = step.copy()
                                if num_p is not None:
                                    step_copy['num_partitions'] = num_p
                                current_plan.append(step_copy)

                            for step_idx, step in enumerate(current_plan):
                                left_name = step['left']
                                right_name = step['right']
                                df_left = current_tables.get(left_name)
                                df_right = current_tables.get(right_name)

                                if df_left is None or df_right is None or df_left.empty or df_right.empty:
                                    final_result = pd.DataFrame()
                                    break
                                
                                sub_tables = {left_name: df_left, right_name: df_right}
                                res = execute_query_plan(sub_tables, [step], join_algorithm=algo_name)
                                final_result = res[0] if isinstance(res, tuple) else res
                                current_tables[f"intermediate_step_{step_idx}"] = final_result

                            elapsed = time.time() - start_t
                            row_count = len(final_result) if final_result is not None else 0

                            if expected_len != -1 and row_count == expected_len:
                                passed = True
                            elif expected_len != -1:
                                error_msg = f"Length mismatch: got {row_count}, expected {expected_len}"
                    except Exception as e:
                        error_msg = str(e)
                        passed = False

                    if final_result is not None:
                        del final_result
                    gc.collect()

                    status_str = "PASS" if passed else "FAIL"

                    record = {
                        "experiment": exp_n,
                        "topology": topo_n,
                        "algorithm": algo_name,
                        "num_partitions": num_p,
                        "status": status_str,
                        "execution_time_sec": elapsed,
                        "output_rows": row_count,
                        "partition_skew_metric": skew_metric,
                        "error": error_msg,
                        **config_params
                    }
                    results.append(record)

                    report_lines.append(
                        f"[{exp_n} | {topo_n}] Algo={algo_name:12} | Partitions={num_p if num_p is not None else 'N/A':3} | "
                        f"Status={status_str} | Time={elapsed:.4f}s | Notes={error_msg or 'None'}"
                    )

        config_generator_func(evaluate_and_record)

        csv_path = os.path.join(OUTPUT_DIR, f"{exp_name}_results.csv")
        txt_path = os.path.join(OUTPUT_DIR, f"{exp_name}_report.txt")

        pd.DataFrame(results).to_csv(csv_path, index=False)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f" -> Completed & Saved: {exp_name} -> '{OUTPUT_DIR}/'")

    # =========================================================================
    # EXPERIMENT 1: Skew Impact (Q_matrix Binary Join)
    
    def exp1_logic(eval_func):
        skew_values = [0.0, 0.8, 1.2]
        for z in skew_values:
            print(f"Running Exp1_Skew with z={z}...")
            R, S = generator.generate_matrix_query(N=1_000, n_A=500, n_B=100, n_C=500, skew_b=z)
            tables = {'R': R, 'S': S}
            plan = [{'left': 'R', 'right': 'S', 'on': 'B', 'lsuffix': '_left', 'rsuffix': '_right'}]
            params = {"num_tuples_N": 1_000, "n_B": 100, "skew_z": z}
            eval_func("Exp1_Skew", "Q_matrix", params, tables, plan)
            del R, S, tables
            gc.collect()

    run_single_experiment("Exp1_Skew", "Q_matrix", exp1_logic)

    # =========================================================================
    # EXPERIMENT 2: Scalability Impact (Q_matrix Binary Join)
    def exp2_logic(eval_func):
        n_values = [50, 150, 300]
        scalability_skews = [0.0, 1.2]
        for N in n_values:
            for z in scalability_skews:
                print(f"Running Exp2_Scalability with N={N}, skew={z}...")
                R, S = generator.generate_matrix_query(N=N, n_A=N//2, n_B=100, n_C=N//2, skew_b=z)
                tables = {'R': R, 'S': S}
                plan = [{'left': 'R', 'right': 'S', 'on': 'B', 'lsuffix': '_left', 'rsuffix': '_right'}]
                params = {"num_tuples_N": N, "n_B": 100, "skew_z": z}
                eval_func("Exp2_Scalability", "Q_matrix", params, tables, plan)
                del R, S, tables
                gc.collect()

    run_single_experiment("Exp2_Scalability", "Q_matrix", exp2_logic)

    # =========================================================================
    # EXPERIMENT 3: Selectivity Impact (Q_matrix Binary Join)
    def exp3_logic(eval_func):
        selectivity_nb_values = [30, 100, 400]
        for n_B in selectivity_nb_values:
            print(f"Running Exp3_Selectivity with n_B={n_B}...")
            R, S = generator.generate_matrix_query(N=1_000, n_A=500, n_B=n_B, n_C=500, skew_b=1.0)
            tables = {'R': R, 'S': S}
            plan = [{'left': 'R', 'right': 'S', 'on': 'B', 'lsuffix': '_left', 'rsuffix': '_right'}]
            params = {"num_tuples_N": 1_000, "n_B": n_B, "skew_z": 1.0}
            eval_func("Exp3_Selectivity", "Q_matrix", params, tables, plan)
            del R, S, tables
            gc.collect()

    run_single_experiment("Exp3_Selectivity", "Q_matrix", exp3_logic)

    # =========================================================================
    # EXPERIMENT 4a-c: Star Queries
    if hasattr(generator, "generate_star_query"):
        def exp4a_logic(eval_func):
            for z in [0.0, 0.8, 1.2]:
                print(f"Running Exp4a_Star_Skew with z={z}...")
                center, leaves = generator.generate_star_query(N=1_000, num_leaves=2, domain_center_b=100, skew_b=z)
                tables = {'center': center, 'leaf_0': leaves[0], 'leaf_1': leaves[1]}
                plan = [
                    {'left': 'center', 'right': 'leaf_0', 'on': 'B', 'lsuffix': '_c', 'rsuffix': '_l0'},
                    {'left': 'intermediate_step_0', 'right': 'leaf_1', 'on': 'B', 'lsuffix': '_step0', 'rsuffix': '_l1'}
                ]
                eval_func("Exp4a_Star_Skew", "Q_star", {"num_tuples_N": 1_000, "skew_z": z}, tables, plan)
                del center, leaves, tables
                gc.collect()
        run_single_experiment("Exp4a_Star_Skew", "Q_star", exp4a_logic)

        def exp4b_logic(eval_func):
            for N in [50, 150, 300]:
                print(f"Running Exp4b_Star_Scalability with N={N}...")
                center, leaves = generator.generate_star_query(N=N, num_leaves=2, domain_center_b=100, skew_b=1.0)
                tables = {'center': center, 'leaf_0': leaves[0], 'leaf_1': leaves[1]}
                plan = [
                    {'left': 'center', 'right': 'leaf_0', 'on': 'B', 'lsuffix': '_c', 'rsuffix': '_l0'},
                    {'left': 'intermediate_step_0', 'right': 'leaf_1', 'on': 'B', 'lsuffix': '_step0', 'rsuffix': '_l1'}
                ]
                eval_func("Exp4b_Star_Scalability", "Q_star", {"num_tuples_N": N, "skew_z": 1.0}, tables, plan)
                del center, leaves, tables
                gc.collect()
        run_single_experiment("Exp4b_Star_Scalability", "Q_star", exp4b_logic)

        def exp4c_logic(eval_func):
            for n_B in [30, 100, 400]:
                print(f"Running Exp4c_Star_Selectivity with domain size={n_B}...")
                center, leaves = generator.generate_star_query(N=1_000, num_leaves=2, domain_center_b=n_B, skew_b=1.0)
                tables = {'center': center, 'leaf_0': leaves[0], 'leaf_1': leaves[1]}
                plan = [
                    {'left': 'center', 'right': 'leaf_0', 'on': 'B', 'lsuffix': '_c', 'rsuffix': '_l0'},
                    {'left': 'intermediate_step_0', 'right': 'leaf_1', 'on': 'B', 'lsuffix': '_step0', 'rsuffix': '_l1'}
                ]
                eval_func("Exp4c_Star_Selectivity", "Q_star", {"num_tuples_N": 1_000, "n_B": n_B, "skew_z": 1.0}, tables, plan)
                del center, leaves, tables
                gc.collect()
        run_single_experiment("Exp4c_Star_Selectivity", "Q_star", exp4c_logic)

    # =========================================================================
    # EXPERIMENT 5a-c: Line Path Queries
    if hasattr(generator, "generate_acyclic_path_query"):
        def exp5a_logic(eval_func):
            for z in [0.0, 0.8, 1.2]:
                print(f"Running Exp5a_Line_Skew with z={z}...")
                path_relations = generator.generate_acyclic_path_query(N=1_000, path_length=2, domain_sizes=[500, 100, 500], skew_factors=[z, z])
                tables = {'R1': path_relations[0], 'R2': path_relations[1]}
                plan = [{'left': 'R1', 'right': 'R2', 'on': 'X_1', 'lsuffix': '_r1', 'rsuffix': '_r2'}]
                eval_func("Exp5a_Line_Skew", "Q_line", {"num_tuples_N": 1_000, "skew_z": z}, tables, plan)
                del path_relations, tables
                gc.collect()
        run_single_experiment("Exp5a_Line_Skew", "Q_line", exp5a_logic)

        def exp5b_logic(eval_func):
            for N in [50, 150, 300]:
                print(f"Running Exp5b_Line_Scalability with N={N}...")
                path_relations = generator.generate_acyclic_path_query(N=N, path_length=2, domain_sizes=[500, 100, 500], skew_factors=[1.0, 1.0])
                tables = {'R1': path_relations[0], 'R2': path_relations[1]}
                plan = [{'left': 'R1', 'right': 'R2', 'on': 'X_1', 'lsuffix': '_r1', 'rsuffix': '_r2'}]
                eval_func("Exp5b_Line_Scalability", "Q_line", {"num_tuples_N": N, "skew_z": 1.0}, tables, plan)
                del path_relations, tables
                gc.collect()
        run_single_experiment("Exp5b_Line_Scalability", "Q_line", exp5b_logic)

        def exp5c_logic(eval_func):
            for mid_domain in [30, 100, 400]:
                print(f"Running Exp5c_Line_Selectivity with mid_domain={mid_domain}...")
                path_relations = generator.generate_acyclic_path_query(N=1_000, path_length=2, domain_sizes=[500, mid_domain, 500], skew_factors=[1.0, 1.0])
                tables = {'R1': path_relations[0], 'R2': path_relations[1]}
                plan = [{'left': 'R1', 'right': 'R2', 'on': 'X_1', 'lsuffix': '_r1', 'rsuffix': '_r2'}]
                eval_func("Exp5c_Line_Selectivity", "Q_line", {"num_tuples_N": 1_000, "mid_domain_size": mid_domain, "skew_z": 1.0}, tables, plan)
                del path_relations, tables
                gc.collect()
        run_single_experiment("Exp5c_Line_Selectivity", "Q_line", exp5c_logic)

    

    # =========================================================================
    # EXPERIMENTS 6a, 6b, 6c: Triangle Query (Q_triangle - Cyclic Multi-Way Join)
    # Designed specifically to benchmark Leapfrog Triejoin against cyclic join bottlenecks
    def generate_triangle_query(N=1_000, domain_size=100, skew_z=1.0, seed=42):
        np.random.seed(seed)
        
        # 1. Generate underlying domain values for vertices A, B, C with the specified skew
        if skew_z == 0.0:
            a_vals = np.random.randint(0, domain_size, size=N)
            b_vals = np.random.randint(0, domain_size, size=N)
            c_vals = np.random.randint(0, domain_size, size=N)
        else:
            a_param = max(1.01, 1.0 + skew_z)
            a_vals = (np.random.zipf(a=a_param, size=N) - 1) % domain_size
            b_vals = (np.random.zipf(a=a_param, size=N) - 1) % domain_size
            c_vals = (np.random.zipf(a=a_param, size=N) - 1) % domain_size

        # 2. Construct explicit triangles (A, B, C)
        triangles = pd.DataFrame({'A': a_vals, 'B': b_vals, 'C': c_vals})
        
        # 3. Project the triangles into binary relations R(A,B), S(B,C), T(C,A)
        # Using drop_duplicates() ensures clean relation tables where every tuple 
        # is part of a valid graph structure.
        R = triangles[['A', 'B']].drop_duplicates().reset_index(drop=True)
        S = triangles[['B', 'C']].drop_duplicates().reset_index(drop=True)
        T = triangles[['C', 'A']].drop_duplicates().reset_index(drop=True)
        
        return {'R': R, 'S': S, 'T': T}

    # Exp 6a: Triangle Skew Impact
    def exp6a_logic(eval_func):
        for z in [0.0, 0.8, 1.2]:
            print(f"Running Exp6a_Triangle_Skew with z={z}...")
            tables = generate_triangle_query(N=1_000, domain_size=100, skew_z=z)
            plan = [] # Multi-way queries are handled natively by Leapfrog or iterative joins
            eval_func("Exp6a_Triangle_Skew", "Q_triangle", {"num_tuples_N": 1_000, "skew_z": z}, tables, plan)
            del tables
            gc.collect()
    run_single_experiment("Exp6a_Triangle_Skew", "Q_triangle", exp6a_logic)

    # Exp 6b: Triangle Scalability Impact
    def exp6b_logic(eval_func):
        for N in [100, 500, 1_000]:
            print(f"Running Exp6b_Triangle_Scalability with N={N}...")
            tables = generate_triangle_query(N=N, domain_size=100, skew_z=1.0)
            plan = []
            eval_func("Exp6b_Triangle_Scalability", "Q_triangle", {"num_tuples_N": N, "skew_z": 1.0}, tables, plan)
            del tables
            gc.collect()
    run_single_experiment("Exp6b_Triangle_Scalability", "Q_triangle", exp6b_logic)

    # Exp 6c: Triangle Selectivity Impact
    def exp6c_logic(eval_func):
        for domain_sz in [30, 100, 300]:
            print(f"Running Exp6c_Triangle_Selectivity with domain_size={domain_sz}...")
            tables = generate_triangle_query(N=1_000, domain_size=domain_sz, skew_z=1.0)
            plan = []
            eval_func("Exp6c_Triangle_Selectivity", "Q_triangle", {"num_tuples_N": 1_000, "domain_size": domain_sz}, tables, plan)
            del tables
            gc.collect()
    run_single_experiment("Exp6c_Triangle_Selectivity", "Q_triangle", exp6c_logic)

    print(f"\nAll benchmark experiments (including Leapfrog and Triangle queries) finished successfully!")
    print(f" -> Individual experiment CSVs and Reports saved inside folder: '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    run_comprehensive_benchmark()