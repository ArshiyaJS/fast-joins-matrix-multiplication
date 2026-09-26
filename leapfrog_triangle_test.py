import pandas as pd
import numpy as np
from src.baselines.leapfrog_generalised import RobustGeneralLFTJEngine

def run_edge_case_tests():
    edge_test_cases = []

    # ==========================================
    # 1. BINARY JOIN EDGE CASES (2 Relations)
    # ==========================================
    
    # Edge Case 1.1: Completely Empty Relation
    binary_empty = [
        {'name': 'R', 'table': pd.DataFrame({'A': [1, 2], 'B': [10, 20]}), 'attrs': ['A', 'B']},
        {'name': 'S', 'table': pd.DataFrame(columns=['B', 'C']), 'attrs': ['B', 'C']} # Empty table
    ]
    edge_test_cases.append(("Binary: Empty Relation", binary_empty, 0))

    # Edge Case 1.2: Zero Overlap (Disjoint Values)
    binary_disjoint = [
        {'name': 'R', 'table': pd.DataFrame({'A': [1, 2], 'B': [10, 20]}), 'attrs': ['A', 'B']},
        {'name': 'S', 'table': pd.DataFrame({'B': [99, 100], 'C': [500, 600]}), 'attrs': ['B', 'C']}
    ]
    edge_test_cases.append(("Binary: Disjoint Values (No Match)", binary_disjoint, 0))


    # ==========================================
    # 2. LINE QUERY EDGE CASES
    # ==========================================
    
    # Edge Case 2.1: Broken Pipeline (Middle relation blocks flow)
    line_broken = [
        {'name': 'R', 'table': pd.DataFrame({'A': [1], 'B': [10]}), 'attrs': ['A', 'B']},
        {'name': 'S', 'table': pd.DataFrame({'B': [10], 'C': [999]}), 'attrs': ['B', 'C']}, # C doesn't match T
        {'name': 'T', 'table': pd.DataFrame({'C': [100], 'D': [1000]}), 'attrs': ['C', 'D']}
    ]
    edge_test_cases.append(("Line: Broken Middle Link", line_broken, 0))


    # ==========================================
    # 3. STAR QUERY EDGE CASES ($q^*$)
    # ==========================================
    
    # Edge Case 3.1: Duplicate Rows in Satellite Relations
    star_duplicates = [
        {'name': 'R', 'table': pd.DataFrame({'A': [1, 1, 2], 'B': [10, 10, 20]}), 'attrs': ['A', 'B']}, # Duplicate rows for A=1
        {'name': 'S', 'table': pd.DataFrame({'A': [1, 1, 2], 'C': [100, 100, 200]}), 'attrs': ['A', 'C']},
        {'name': 'T', 'table': pd.DataFrame({'A': [1, 2], 'D': [1000, 2000]}), 'attrs': ['A', 'D']}
    ]
    edge_test_cases.append(("Star: Duplicate Rows Handling", star_duplicates, 2)) # Should handle gracefully and yield unique tuples


    # ==========================================
    # 4. TRIANGLE QUERY EDGE CASES (Cyclic)
    # ==========================================
    
    # Edge Case 4.1: Sub-triangle with dangling chains (only 1 valid triangle instead of multiple)
    triangle_sparse = [
        {'name': 'R', 'table': pd.DataFrame({'A': [1, 5], 'B': [1, 5]}), 'attrs': ['A', 'B']},
        {'name': 'S', 'table': pd.DataFrame({'B': [1, 8], 'C': [10, 80]}), 'attrs': ['B', 'C']},
        {'name': 'T', 'table': pd.DataFrame({'C': [10, 90], 'A': [1, 9]}), 'attrs': ['C', 'A']}
    ]
    edge_test_cases.append(("Triangle: Sparse/Single Closed Triangle", triangle_sparse, 1))

    # Edge Case 4.2: Fully Disconnected Triangle Variables (Zero matches)
    triangle_zero = [
        {'name': 'R', 'table': pd.DataFrame({'A': [1], 'B': [1]}), 'attrs': ['A', 'B']},
        {'name': 'S', 'table': pd.DataFrame({'B': [2], 'C': [2]}), 'attrs': ['B', 'C']},
        {'name': 'T', 'table': pd.DataFrame({'C': [3], 'A': [3]}), 'attrs': ['C', 'A']}
    ]
    edge_test_cases.append(("Triangle: Complete Mismatch (0 Triangles)", triangle_zero, 0))


    # ==========================================
    # 5. 4-CYCLE QUERY EDGE CASES ($q$)
    # ==========================================
    
    # Edge Case 5.1: Constant / Single-Value Columns across a Cycle
    cycle_constants = [
        {'name': 'R', 'table': pd.DataFrame({'A': [1, 1], 'B': [5, 5]}), 'attrs': ['A', 'B']},
        {'name': 'S', 'table': pd.DataFrame({'B': [5, 5], 'C': [10, 10]}), 'attrs': ['B', 'C']},
        {'name': 'T', 'table': pd.DataFrame({'C': [10, 10], 'D': [20, 20]}), 'attrs': ['C', 'D']},
        {'name': 'U', 'table': pd.DataFrame({'D': [20, 20], 'A': [1, 1]}), 'attrs': ['D', 'A']}
    ]
    edge_test_cases.append(("4-Cycle: Constant Value Loops", cycle_constants, 1))


    # ==========================================
    # EXECUTION HARNESS
    # ==========================================
    print("==================================================")
    print("      ROBUST GENERAL LFTJ EDGE CASE SUITE         ")
    print("==================================================")

    passed_count = 0
    for name, query_def, expected_rows in edge_test_cases:
        print(f"\n[TEST] {name}")
        try:
            engine = RobustGeneralLFTJEngine(query_def)
            output_df, diagnostics = engine.execute()
            actual_rows = diagnostics['output_rows']
            
            status = "PASS" if actual_rows == expected_rows else "FAIL"
            if status == "PASS":
                passed_count += 1
                
            print(f"  -> Order: {engine.variable_order}")
            print(f"  -> Expected Rows: {expected_rows} | Actual Rows: {actual_rows} [{status}]")
            if actual_rows > 0:
                print("  -> Result snippet:")
                print(output_df.head(2).to_string(index=False))
        except Exception as e:
            print(f"  -> CRASHED WITH EXCEPTION: {e}")
        print("-" * 50)

    print(f"\nEdge Case Summary: {passed_count}/{len(edge_test_cases)} tests passed successfully.")

if __name__ == "__main__":
    run_edge_case_tests()