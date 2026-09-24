# run_comprehensive_benchmark.py
import time
import gc
import pandas as pd
import numpy as np
import os

from src.encoders.workload_generator_zipfianSkewSelectivity import MultiTopologyWorkloadGenerator
from src.joins.query_executor_combined import execute_query_plan, JOIN_ALGORITHMS

# Create output directory for modular results
OUTPUT_DIR = "benchmark results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_comprehensive_benchmark():
    generator = MultiTopologyWorkloadGenerator(base_seed=42)
    algorithms = ["simple_hash", "grace_hash", "hybrid_hash", "radix_hash", "sort_merge", "pandas"]

    def run_single_experiment(exp_name, topology_name, config_generator_func):
        results = []
        report_lines = []

        report_lines.append("============================================================")
        report_lines.append(f"    BENCHMARK REPORT: {exp_name} ({topology_name})        ")
        report_lines.append("============================================================\n")

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
                if algo_name != "pandas" and algo_name not in JOIN_ALGORITHMS:
                    continue

                # Define partition configurations based on algorithm capability
                if algo_name in ["grace_hash", "radix_hash", "hybrid_hash"]:
                    partition_configs = [4, 8, 16]
                else:
                    partition_configs = [None]  # Partition-less algorithms run once

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
        n_values = [50, 1_50, 3_00]
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
    # EXPERIMENT 4a: Skew Impact (Q_star)
    if hasattr(generator, "generate_star_query"):
        def exp4a_logic(eval_func):
            star_skews = [0.0, 0.8, 1.2]
            for z in star_skews:
                print(f"Running Exp4a_Star_Skew with z={z}...")
                center, leaves = generator.generate_star_query(N=1_000, num_leaves=2, domain_center_b=100, skew_b=z)
                tables = {'center': center, 'leaf_0': leaves[0], 'leaf_1': leaves[1]}
                plan = [
                    {'left': 'center', 'right': 'leaf_0', 'on': 'B', 'lsuffix': '_c', 'rsuffix': '_l0'},
                    {'left': 'intermediate_step_0', 'right': 'leaf_1', 'on': 'B', 'lsuffix': '_step0', 'rsuffix': '_l1'}
                ]
                params = {"num_tuples_N": 1_000, "skew_z": z}
                eval_func("Exp4a_Star_Skew", "Q_star", params, tables, plan)
                del center, leaves, tables
                gc.collect()

        run_single_experiment("Exp4a_Star_Skew", "Q_star", exp4a_logic)

        def exp4b_logic(eval_func):
            star_n_values = [50, 1_50, 3_00]
            for N in star_n_values:
                print(f"Running Exp4b_Star_Scalability with N={N}...")
                center, leaves = generator.generate_star_query(N=N, num_leaves=2, domain_center_b=100, skew_b=1.0)
                tables = {'center': center, 'leaf_0': leaves[0], 'leaf_1': leaves[1]}
                plan = [
                    {'left': 'center', 'right': 'leaf_0', 'on': 'B', 'lsuffix': '_c', 'rsuffix': '_l0'},
                    {'left': 'intermediate_step_0', 'right': 'leaf_1', 'on': 'B', 'lsuffix': '_step0', 'rsuffix': '_l1'}
                ]
                params = {"num_tuples_N": N, "skew_z": 1.0}
                eval_func("Exp4b_Star_Scalability", "Q_star", params, tables, plan)
                del center, leaves, tables
                gc.collect()

        run_single_experiment("Exp4b_Star_Scalability", "Q_star", exp4b_logic)

        def exp4c_logic(eval_func):
            star_nb_values = [30, 100, 400]
            for n_B in star_nb_values:
                print(f"Running Exp4c_Star_Selectivity with domain size={n_B}...")
                center, leaves = generator.generate_star_query(N=1_000, num_leaves=2, domain_center_b=n_B, skew_b=1.0)
                tables = {'center': center, 'leaf_0': leaves[0], 'leaf_1': leaves[1]}
                plan = [
                    {'left': 'center', 'right': 'leaf_0', 'on': 'B', 'lsuffix': '_c', 'rsuffix': '_l0'},
                    {'left': 'intermediate_step_0', 'right': 'leaf_1', 'on': 'B', 'lsuffix': '_step0', 'rsuffix': '_l1'}
                ]
                params = {"num_tuples_N": 1_000, "n_B": n_B, "skew_z": 1.0}
                eval_func("Exp4c_Star_Selectivity", "Q_star", params, tables, plan)
                del center, leaves, tables
                gc.collect()

        run_single_experiment("Exp4c_Star_Selectivity", "Q_star", exp4c_logic)

    # =========================================================================
    # EXPERIMENT 5a: Skew Impact (Q_line)
    if hasattr(generator, "generate_acyclic_path_query"):
        def exp5a_logic(eval_func):
            line_skews = [0.0, 0.8, 1.2]
            for z in line_skews:
                print(f"Running Exp5a_Line_Skew with z={z}...")
                line_domains = [500, 100, 500]
                path_relations = generator.generate_acyclic_path_query(
                    N=1_000, path_length=2, domain_sizes=line_domains, skew_factors=[z, z]
                )
                tables = {'R1': path_relations[0], 'R2': path_relations[1]}
                plan = [{'left': 'R1', 'right': 'R2', 'on': 'X_1', 'lsuffix': '_r1', 'rsuffix': '_r2'}]
                params = {"num_tuples_N": 1_000, "skew_z": z}
                eval_func("Exp5a_Line_Skew", "Q_line", params, tables, plan)
                del path_relations, tables
                gc.collect()

        run_single_experiment("Exp5a_Line_Skew", "Q_line", exp5a_logic)

        def exp5b_logic(eval_func):
            line_n_values = [50, 1_50, 3_00]
            for N in line_n_values:
                print(f"Running Exp5b_Line_Scalability with N={N}...")
                line_domains = [500, 100, 500]
                path_relations = generator.generate_acyclic_path_query(
                    N=N, path_length=2, domain_sizes=line_domains, skew_factors=[1.0, 1.0]
                )
                tables = {'R1': path_relations[0], 'R2': path_relations[1]}
                plan = [{'left': 'R1', 'right': 'R2', 'on': 'X_1', 'lsuffix': '_r1', 'rsuffix': '_r2'}]
                params = {"num_tuples_N": N, "skew_z": 1.0}
                eval_func("Exp5b_Line_Scalability", "Q_line", params, tables, plan)
                del path_relations, tables
                gc.collect()

        run_single_experiment("Exp5b_Line_Scalability", "Q_line", exp5b_logic)

        def exp5c_logic(eval_func):
            for mid_domain in [30, 100, 400]: 
                print(f"Running Exp5c_Line_Selectivity with mid_domain={mid_domain}...")
                line_domains = [500, mid_domain, 500]
                path_relations = generator.generate_acyclic_path_query(
                    N=1_000, path_length=2, domain_sizes=line_domains, skew_factors=[1.0, 1.0]
                )
                tables = {'R1': path_relations[0], 'R2': path_relations[1]}
                plan = [{'left': 'R1', 'right': 'R2', 'on': 'X_1', 'lsuffix': '_r1', 'rsuffix': '_r2'}]
                params = {"num_tuples_N": 1_000, "mid_domain_size": mid_domain, "skew_z": 1.0}
                eval_func("Exp5c_Line_Selectivity", "Q_line", params, tables, plan)
                del path_relations, tables
                gc.collect()

        run_single_experiment("Exp5c_Line_Selectivity", "Q_line", exp5c_logic)

    print(f"\nAll benchmark experiments finished successfully!")
    print(f" -> Individual experiment CSVs and Reports saved inside folder: '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    run_comprehensive_benchmark()