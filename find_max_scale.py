# find_max_scale.py
import gc
import sys
import time
try:
    import psutil
except ImportError:
    psutil = None

from src.encoders.workload_generator_zipfianSkewSelectivity import MultiTopologyWorkloadGenerator
from src.joins.query_executor_combined import execute_query_plan

def check_system_memory():
    if psutil:
        mem = psutil.virtual_memory()
        return mem.percent, mem.available / (1024 ** 3) # in GB
    return None, None

def probe_max_tuple_size():
    generator = MultiTopologyWorkloadGenerator(base_seed=42)
    
    # Test increments
    test_sizes = [10_000, 50_000, 100_000, 250_000, 500_000, 750_000, 1_000_000]
    
    print("==================================================")
    print("      LAPTOP CAPACITY & MAX SCALE PROBE         ")
    print("==================================================\n")
    
    if psutil:
        init_mem, init_avail = check_system_memory()
        print(f"Initial System RAM Used: {init_mem}% | Available: {init_avail:.2f} GB\n")

    max_successful_N = 0

    for N in test_sizes:
        print(f"Probing N = {N:,} tuples...", end=" ", flush=True)
        start_time = time.time()
        
        try:
            # Generate workload tables
            R, S = generator.generate_matrix_query(N=N, n_A=N//2, n_B=1000, n_C=N//2, skew_b=1.2)
            tables = {'R': R, 'S': S}
            plan = [{'left': 'R', 'right': 'S', 'on': 'B', 'lsuffix': '_left', 'rsuffix': '_right'}]
            
            # Run a heavy algorithm (e.g., pandas or grace_hash) to test peak allocation
            result_df = execute_query_plan(tables, plan, join_algorithm="pandas")
            
            elapsed = time.time() - start_time
            max_successful_N = N
            
            mem_pct, avail_gb = check_system_memory()
            mem_str = f" | RAM Used: {mem_pct}% ({avail_gb:.2f} GB free)" if psutil else ""
            print(f"SUCCESS in {elapsed:.2f}s{mem_str}")
            
            # Clean up memory
            del R, S, tables, result_df
            gc.collect()
            
        except MemoryError:
            print("\n[!] CRITICAL: Python raised a MemoryError.")
            break
        except Exception as e:
            print(f"\n[!] FAILED with error: {e}")
            break

    print("\n==================================================")
    print(f" MAXIMUM SAFE TUPLE SIZE (N): {max_successful_N:,}")
    print("==================================================")
    if max_successful_N >= 500_000:
        print("Your laptop is powerful enough for large-scale benchmarks!")
    elif max_successful_N >= 100_000:
        print("Your laptop handles medium workloads well. Stick to N <= 100k for heavy join loops.")
    else:
        print("Your laptop has limited RAM. Keep your benchmark sizes small (N <= 50k) and keep `gc.collect()` enabled.")

if __name__ == "__main__":
    probe_max_tuple_size()