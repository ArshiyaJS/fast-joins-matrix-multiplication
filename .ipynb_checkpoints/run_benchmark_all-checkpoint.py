# run_comprehensive_benchmark.py
import time
import pandas as pd
import numpy as np
import os

from src.encoders.workload_generator_zipfianSkewSelectivity import MultiTopologyWorkloadGenerator
from src.joins.query_executor_combined import execute_query_plan, JOIN_ALGORITHMS

REPORT_TXT = "comprehensive_benchmark_report.txt"
REPORT_CSV = "comprehensive_benchmark_results.csv"

def run_comprehensive_benchmark():
    generator = MultiTopologyWorkloadGenerator(base_seed=42)
    # Focus list or full set from JOIN_ALGORITHMS
    algorithms = ["simple_hash", "grace_hash", "hybrid_hash", "radix_hash", "sort_merge", "pandas"]

    results = []
    report_lines = []

    report_lines.append("============================================================")
    report_lines.append("      COMPREHENSIVE MULTI-TOPOLOGY JOIN BENCHMARK REPORT    ")
    report_lines.append("============================================================\n")

    def evaluate_and_record(exp_name, topology_name, config_params, tables, plan):
        """
        Generic helper for evaluating binary or multi-step query plans across 
        all algorithms, with proper algorithm propagation and validation.
        """
        # 1. Compute Ground Truth using Pandas baseline
        try:
            ground_truth_df = execute_query_plan(tables.copy(), plan, join_algorithm="pandas")
            expected_len = len(ground_truth_df)
        except Exception as e:
            report_lines.append(f"[{exp_name} | {topology_name}] [PANDAS BASELINE FAILED]: {e}")
            expected_len = -1

        # 2. Iterate through candidate algorithms and propagate correctly
        for algo_name in algorithms:
            if not algo_name == "pandas" and algo_name not in JOIN_ALGORITHMS:
                continue

            # Handle algorithm-specific parameter grids (e.g. partition counts)
            partition_configs = [32]
            if algo_name in ["grace_hash", "radix_hash"]:
                partition_configs = [8, 32, 128]

            for num_p in partition_configs:
                start_t = time.time()
                passed = False
                error_msg = ""
                row_count = 0
                skew_metric = 0.0
                elapsed = 0.0
                final_result = None

                try:
                    if algo_name == "pandas":
                        # Time pandas execution accurately
                        ref_start = time.time()
                        final_result = execute_query_plan(tables.copy(), plan, join_algorithm="pandas")
                        elapsed = time.time() - ref_start
                        row_count = expected_len
                        passed = True
                    else:
                        # Propagate chosen join algorithm and partition options down to execution steps
                        start_t = time.time()
                        try:
                            final_result = execute_query_plan(
                                tables.copy(), 
                                plan, 
                                join_algorithm=algo_name, 
                                num_partitions=num_p
                            )
                        except TypeError:
                            # Fallback if executor plan signature does not accept num_partitions directly
                            final_result = execute_query_plan(
                                tables.copy(), 
                                plan, 
                                join_algorithm=algo_name
                            )
                        
                        elapsed = time.time() - start_t
                        row_count = len(final_result) if final_result is not None else 0

                        if expected_len != -1 and row_count == expected_len:
                            passed = True
                        elif expected_len != -1:
                            error_msg = f"Length mismatch: got {row_count}, expected {expected_len}"
                except Exception as e:
                    error_msg = str(e)
                    passed = False

                status_str = "PASS" if passed else "FAIL"

                record = {
                    "experiment": exp_name,
                    "topology": topology_name,
                    "algorithm": algo_name,
                    "num_partitions": num_p if algo_name in ["grace_hash", "radix_hash"] else None,
                    "status": status_str,
                    "execution_time_sec": elapsed,
                    "output_rows": row_count,
                    "partition_skew_metric": skew_metric,
                    "error": error_msg,
                    **config_params
                }
                results.append(record)

                report_lines.append(
                    f"[{exp_name} | {topology_name}] Algo={algo_name:12} | Partitions={num_p if algo_name in ['grace_hash', 'radix_hash'] else 'N/A':3} | "
                    f"Status={status_str} | Time={elapsed:.4f}s | Notes={error_msg or 'None'}"
                )

    # =========================================================================
    # EXPERIMENT 1: Skew Impact (Q_matrix Binary Join)
    report_lines.append("\n--- EXPERIMENT 1: Skew Impact Analysis (Q_matrix) ---")
    skew_values = [0.0, 0.8, 1.2, 1.5]
    for z in skew_values:
        print(f"Running Experiment 1 [Skew Impact] with z={z}...")
        R, S = generator.generate_matrix_query(N=50_000, n_A=20_000, n_B=1_000, n_C=20_000, skew_b=z)
        tables = {'R': R, 'S': S}
        plan = [{'left': 'R', 'right': 'S', 'on': 'B', 'lsuffix': '_left', 'rsuffix': '_right'}]
        params = {"num_tuples_N": 50_000, "n_B": 1_000, "skew_z": z}
        evaluate_and_record("Exp1_Skew", "Q_matrix", params, tables, plan)

    # =========================================================================
    # EXPERIMENT 2: Scalability Impact (Q_matrix Binary Join)
    report_lines.append("\n--- EXPERIMENT 2: Scalability Impact Analysis (Q_matrix) ---")
    n_values = [10_000, 100_000, 500_000]
    scalability_skews = [0.0, 1.2]
    for N in n_values:
        for z in scalability_skews:
            print(f"Running Experiment 2 [Scalability] with N={N}, skew={z}...")
            R, S = generator.generate_matrix_query(N=N, n_A=N//2, n_B=1_000, n_C=N//2, skew_b=z)
            tables = {'R': R, 'S': S}
            plan = [{'left': 'R', 'right': 'S', 'on': 'B', 'lsuffix': '_left', 'rsuffix': '_right'}]
            params = {"num_tuples_N": N, "n_B": 1_000, "skew_z": z}
            evaluate_and_record("Exp2_Scalability", "Q_matrix", params, tables, plan)

    # =========================================================================
    # EXPERIMENT 3: Selectivity Impact (Q_matrix Binary Join)
    report_lines.append("\n--- EXPERIMENT 3: Selectivity Impact Analysis (Q_matrix) ---")
    selectivity_nb_values = [100, 1_000, 10_000]
    for n_B in selectivity_nb_values:
        print(f"Running Experiment 3 [Selectivity] with n_B={n_B}...")
        R, S = generator.generate_matrix_query(N=50_000, n_A=20_000, n_B=n_B, n_C=20_000, skew_b=1.0)
        tables = {'R': R, 'S': S}
        plan = [{'left': 'R', 'right': 'S', 'on': 'B', 'lsuffix': '_left', 'rsuffix': '_right'}]
        params = {"num_tuples_N": 50_000, "n_B": n_B, "skew_z": 1.0}
        evaluate_and_record("Exp3_Selectivity", "Q_matrix", params, tables, plan)

    # =========================================================================
    # EXPERIMENTS 4a, 4b, 4c: Star Joins (Q_star)
    if hasattr(generator, "generate_star_query"):
        
        # EXPERIMENT 4a: Skew Impact (Q_star)
        report_lines.append("\n--- EXPERIMENT 4a: Skew Impact (Q_star) ---")
        star_skews = [0.0, 0.8, 1.2, 1.5]
        for z in star_skews:
            print(f"Running Experiment 4a [Star Skew] with z={z}...")
            center, leaves = generator.generate_star_query(N=50_000, num_leaves=2, domain_center_b=1_000, skew_b=z)
            tables = {'center': center, 'leaf_0': leaves[0], 'leaf_1': leaves[1]}
            plan = [
                {'left': 'center', 'right': 'leaf_0', 'on': 'B', 'lsuffix': '_c', 'rsuffix': '_l0'},
                {'left': 'intermediate_step_0', 'right': 'leaf_1', 'on': 'B', 'lsuffix': '_step0', 'rsuffix': '_l1'}
            ]
            params = {"num_tuples_N": 50_000, "skew_z": z}
            evaluate_and_record("Exp4a_Star_Skew", "Q_star", params, tables, plan)

        # EXPERIMENT 4b: Scalability Impact (Q_star)
        report_lines.append("\n--- EXPERIMENT 4b: Scalability Impact (Q_star) ---")
        star_n_values = [10_000, 100_000, 500_000]
        for N in star_n_values:
            print(f"Running Experiment 4b [Star Scalability] with N={N}...")
            center, leaves = generator.generate_star_query(N=N, num_leaves=2, domain_center_b=1_000, skew_b=1.0)
            tables = {'center': center, 'leaf_0': leaves[0], 'leaf_1': leaves[1]}
            plan = [
                {'left': 'center', 'right': 'leaf_0', 'on': 'B', 'lsuffix': '_c', 'rsuffix': '_l0'},
                {'left': 'intermediate_step_0', 'right': 'leaf_1', 'on': 'B', 'lsuffix': '_step0', 'rsuffix': '_l1'}
            ]
            params = {"num_tuples_N": N, "skew_z": 1.0}
            evaluate_and_record("Exp4b_Star_Scalability", "Q_star", params, tables, plan)

        # EXPERIMENT 4c: Selectivity Impact (Q_star)
        report_lines.append("\n--- EXPERIMENT 4c: Selectivity Impact (Q_star) ---")
        star_nb_values = [100, 1_000, 10_000]
        for n_B in star_nb_values:
            print(f"Running Experiment 4c [Star Selectivity] with domain size={n_B}...")
            center, leaves = generator.generate_star_query(N=50_000, num_leaves=2, domain_center_b=n_B, skew_b=1.0)
            tables = {'center': center, 'leaf_0': leaves[0], 'leaf_1': leaves[1]}
            plan = [
                {'left': 'center', 'right': 'leaf_0', 'on': 'B', 'lsuffix': '_c', 'rsuffix': '_l0'},
                {'left': 'intermediate_step_0', 'right': 'leaf_1', 'on': 'B', 'lsuffix': '_step0', 'rsuffix': '_l1'}
            ]
            params = {"num_tuples_N": 50_000, "n_B": n_B, "skew_z": 1.0}
            evaluate_and_record("Exp4c_Star_Selectivity", "Q_star", params, tables, plan)

    # =========================================================================
    # EXPERIMENTS 5a, 5b, 5c: Acyclic / Line Joins (Q_line)
    if hasattr(generator, "generate_acyclic_path_query"):
        
        # EXPERIMENT 5a: Skew Impact (Q_line)
        report_lines.append("\n--- EXPERIMENT 5a: Skew Impact (Q_line) ---")
        line_skews = [0.0, 0.8, 1.2, 1.5]
        for z in line_skews:
            print(f"Running Experiment 5a [Line Skew] with z={z}...")
            line_domains = [10_000, 1_000, 10_000]
            path_relations = generator.generate_acyclic_path_query(
                N=50_000, path_length=2, domain_sizes=line_domains, skew_factors=[z, z]
            )
            tables = {'R1': path_relations[0], 'R2': path_relations[1]}
            plan = [{'left': 'R1', 'right': 'R2', 'on': 'X_1', 'lsuffix': '_r1', 'rsuffix': '_r2'}]
            params = {"num_tuples_N": 50_000, "skew_z": z}
            evaluate_and_record("Exp5a_Line_Skew", "Q_line", params, tables, plan)

        # EXPERIMENT 5b: Scalability Impact (Q_line)
        report_lines.append("\n--- EXPERIMENT 5b: Scalability Impact (Q_line) ---")
        line_n_values = [10_000, 100_000, 500_000]
        for N in line_n_values:
            print(f"Running Experiment 5b [Line Scalability] with N={N}...")
            line_domains = [10_000, 1_000, 10_000]
            path_relations = generator.generate_acyclic_path_query(
                N=N, path_length=2, domain_sizes=line_domains, skew_factors=[1.0, 1.0]
            )
            tables = {'R1': path_relations[0], 'R2': path_relations[1]}
            plan = [{'left': 'R1', 'right': 'R2', 'on': 'X_1', 'lsuffix': '_r1', 'rsuffix': '_r2'}]
            params = {"num_tuples_N": N, "skew_z": 1.0}
            evaluate_and_record("Exp5b_Line_Scalability", "Q_line", params, tables, plan)

        # EXPERIMENT 5c: Selectivity Impact (Q_line)
        report_lines.append("\n--- EXPERIMENT 5c: Selectivity Impact (Q_line) ---")
        line_mid_domains = [10_000, 100, 10_000] # Smaller middle domain = higher join explosion
        for mid_domain in [100, 1_000, 5_000]:
            print(f"Running Experiment 5c [Line Selectivity] with mid_domain={mid_domain}...")
            line_domains = [10_000, mid_domain, 10_000]
            path_relations = generator.generate_acyclic_path_query(
                N=50_000, path_length=2, domain_sizes=line_domains, skew_factors=[1.0, 1.0]
            )
            tables = {'R1': path_relations[0], 'R2': path_relations[1]}
            plan = [{'left': 'R1', 'right': 'R2', 'on': 'X_1', 'lsuffix': '_r1', 'rsuffix': '_r2'}]
            params = {"num_tuples_N": 50_000, "mid_domain_size": mid_domain, "skew_z": 1.0}
            evaluate_and_record("Exp5c_Line_Selectivity", "Q_line", params, tables, plan)

    # Write out reports to root folder
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    df_results = pd.DataFrame(results)
    df_results.to_csv(REPORT_CSV, index=False)
    
    print(f"\nComprehensive Benchmark completed successfully!")
    print(f" -> Text Report saved to: {REPORT_TXT}")
    print(f" -> CSV Data Matrix saved to: {REPORT_CSV}")

if __name__ == "__main__":
    run_comprehensive_benchmark()