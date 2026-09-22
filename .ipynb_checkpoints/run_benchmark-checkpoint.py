# run_topology_benchmark.py
import time
import pandas as pd
import numpy as np
import os

from src.encoders.workload_generator_zipfianSkewSelectivity import MultiTopologyWorkloadGenerator
from src.joins.query_executor_combined import JOIN_ALGORITHMS

REPORT_TXT = "comprehensive_benchmark_report.txt"
REPORT_CSV = "comprehensive_benchmark_results.csv"

def run_comprehensive_benchmark():
    generator = MultiTopologyWorkloadGenerator(base_seed=42)
    algorithms = ["simple_hash", "grace_hash", "hybrid_hash", "radix_hash", "sort_merge", "pandas"]
    
    results = []
    report_lines = []
    
    report_lines.append("============================================================")
    report_lines.append("      COMPREHENSIVE MULTI-TOPOLOGY JOIN BENCHMARK REPORT    ")
    report_lines.append("============================================================\n")

    def evaluate_and_record(exp_name, config_params, R, S, ref_res, ref_time):
        """Helper to run all algorithms on a given dataset, utilizing pre-computed pandas reference time."""
        expected_len = len(ref_res)

        for algo_name in algorithms:
            join_func = JOIN_ALGORITHMS.get(algo_name)
            if not algo_name == "pandas" and not join_func:
                continue

            # Handle algorithm-specific partition grids
            partition_configs = [32]
            if algo_name in ["grace_hash", "radix_hash"]:
                partition_configs = [8, 32, 128]

            for num_p in partition_configs:
                passed = False
                error_msg = ""
                row_count = 0
                skew_metric = 0.0
                elapsed = 0.0

                try:
                    if algo_name == "pandas":
                        # Use accurate pre-measured reference time
                        elapsed = ref_time
                        row_count = expected_len
                        passed = True
                    else:
                        start_t = time.time()
                        try:
                            res, diagnostics = join_func(R, S, key="B", lsuffix='_left', rsuffix='_right', num_partitions=num_p)
                        except TypeError:
                            res, diagnostics = join_func(R, S, key="B", lsuffix='_left', rsuffix='_right')
                        
                        elapsed = time.time() - start_t
                        row_count = len(res)
                        
                        # Extract partition skew metric from diagnostics
                        p1_sizes = diagnostics.get("df1_partition_sizes", {})
                        if p1_sizes:
                            total_size = sum(p1_sizes.values())
                            max_size = max(p1_sizes.values())
                            skew_metric = max_size / total_size if total_size > 0 else 0.0

                        if row_count == expected_len:
                            passed = True
                        else:
                            error_msg = f"Length mismatch: got {row_count}, expected {expected_len}"
                except Exception as e:
                    error_msg = str(e)
                    passed = False

                status_str = "PASS" if passed else "FAIL"

                record = {
                    "experiment": exp_name,
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
                    f"[{exp_name}] Algo={algo_name:12} | Partitions={num_p if algo_name in ['grace_hash', 'radix_hash'] else 'N/A':3} | "
                    f"SkewMetric={skew_metric:.3f} | Status={status_str} | Time={elapsed:.4f}s | Notes={error_msg if error_msg else 'None'}"
                )

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: Skew Impact (Vary skew_z, Fix N and n_B)
    report_lines.append("\n--- EXPERIMENT 1: Skew Impact Analysis ---")
    skew_values = [0.0, 0.8, 1.2, 1.5]
    for z in skew_values:
        print(f"Running Experiment 1 [Skew Impact] with z={z}...")
        R, S = generator.generate_matrix_query(N=50_000, n_A=20_000, n_B=1_000, n_C=20_000, skew_b=z)
        
        # Measure true pandas execution time
        ref_start = time.time()
        ref_res = pd.merge(R, S, on="B", how="inner")
        ref_time = time.time() - ref_start
        
        params = {"num_tuples_N": 50_000, "n_B": 1_000, "skew_z": z}
        evaluate_and_record("Exp1_Skew", params, R, S, ref_res, ref_time)

    # EXPERIMENT 2: Scalability Impact (Vary N, Test with zero and high skew)
    report_lines.append("\n--- EXPERIMENT 2: Scalability Impact Analysis ---")
    n_values = [10_000, 100_000, 500_000]
    scalability_skews = [0.0, 1.2]
    for N in n_values:
        for z in scalability_skews:
            print(f"Running Experiment 2 [Scalability] with N={N}, skew={z}...")
            R, S = generator.generate_matrix_query(N=N, n_A=N//2, n_B=1_000, n_C=N//2, skew_b=z)
            
            ref_start = time.time()
            ref_res = pd.merge(R, S, on="B", how="inner")
            ref_time = time.time() - ref_start
            
            params = {"num_tuples_N": N, "n_B": 1_000, "skew_z": z}
            evaluate_and_record("Exp2_Scalability", params, R, S, ref_res, ref_time)

    # EXPERIMENT 3: Selectivity Impact (Vary n_B domain size, Fix N and skew)
    report_lines.append("\n--- EXPERIMENT 3: Selectivity / Join Explosion Analysis ---")
    selectivity_nb_values = [100, 1_000, 10_000]
    for n_B in selectivity_nb_values:
        print(f"Running Experiment 3 [Selectivity] with n_B={n_B}...")
        R, S = generator.generate_matrix_query(N=50_000, n_A=20_000, n_B=n_B, n_C=20_000, skew_b=1.0)
        
        ref_start = time.time()
        ref_res = pd.merge(R, S, on="B", how="inner")
        ref_time = time.time() - ref_start
        
        params = {"num_tuples_N": 50_000, "n_B": n_B, "skew_z": 1.0}
        evaluate_and_spec = evaluate_and_record("Exp3_Selectivity", params, R, S, ref_res, ref_time)

    # Write out reports
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    df_results = pd.DataFrame(results)
    df_results.to_csv(REPORT_CSV, index=False)
    
    print(f"\nComprehensive Benchmark completed successfully!")
    print(f" -> Text Report saved to: {REPORT_TXT}")
    print(f" -> CSV Data Matrix saved to: {REPORT_CSV}")

if __name__ == "__main__":
    run_comprehensive_benchmark()