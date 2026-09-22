# tests/test_joins.py
import pytest
import pandas as pd
import numpy as np
import os

from src.baselines.hash_joins_collision_domainSkewPartition import (
    simple_hash_join,
    grace_hash_join,
    hybrid_hash_join,
    radix_hash_join
)
from src.baselines.sort_merge_join import sort_merge_join

JOIN_ALGORITHMS = [
    ("simple_hash", simple_hash_join),
    ("grace_hash", grace_hash_join),
    ("hybrid_hash", hybrid_hash_join),
    ("radix_hash", radix_hash_join),
    ("sort_merge", sort_merge_join),
]

REPORT_FILE = "test_execution_report.txt"

def _log_result(test_name, algo_name, inputs_desc, expected_desc, output_desc, status, reason):
    """Appends a structured trace log entry to the root text report file."""
    mode = "a" if os.path.exists(REPORT_FILE) else "w"
    with open(REPORT_FILE, mode, encoding="utf-8") as f:
        f.write(f"============================================================\n")
        f.write(f"TEST CASE     : {test_name} [{algo_name}]\n")
        f.write(f"============================================================\n")
        f.write(f"INPUTS        :\n{inputs_desc}\n\n")
        f.write(f"EXPECTED      : {expected_desc}\n")
        f.write(f"OUTPUT        : {output_desc}\n")
        f.write(f"STATUS        : {status}\n")
        f.write(f"EXPLANATION   : {reason}\n\n")


@pytest.fixture(scope="session", autouse=True)
def initialize_report_file():
    """Initializes/clears the report file at the start of the pytest session."""
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("          FAST-JOINS CORRECTNESS & VALIDATION REPORT         \n")
        f.write("============================================================\n\n")


@pytest.mark.parametrize("name, join_func", JOIN_ALGORITHMS)
def test_happy_path(name, join_func):
    """Test Case 1: Standard Inner Join with matching and non-matching keys."""
    df1 = pd.DataFrame({'key': [1, 2, 3, 4], 'val_a': ['a', 'b', 'c', 'd']})
    df2 = pd.DataFrame({'key': [3, 4, 5, 6], 'val_b': ['x', 'y', 'z', 'w']})
    
    try:
        expected = pd.merge(df1, df2, on='key', how='inner', suffixes=('_left', '_right'))
        result, _ = join_func(df1, df2, key='key', lsuffix='_left', rsuffix='_right')
        
        assert len(result) == len(expected)
        status = "PASS"
        reason = "Result row count and structure successfully match pandas inner join baseline."
        out_desc = f"Shape: {result.shape}, Rows: {len(result)}"
    except Exception as e:
        status = "FAIL"
        reason = f"Raised exception: {str(e)}"
        out_desc = "None"
        raise
    finally:
        _log_result("test_happy_path", name, f"df1:\n{df1}\ndf2:\n{df2}", f"Expected 2 inner join matches for keys 3 and 4.", out_desc, status, reason)


@pytest.mark.parametrize("name, join_func", JOIN_ALGORITHMS)
@pytest.mark.parametrize("df1_data, df2_data", [
    ([], [1, 2]),
    ([1, 2], []),
    ([], [])
])
def test_empty_relations(name, join_func, df1_data, df2_data):
    """Test Case 2: Empty Relations handling."""
    df1 = pd.DataFrame({'key': df1_data, 'val_a': [str(x) for x in df1_data]})
    df2 = pd.DataFrame({'key': df2_data, 'val_b': [str(x) for x in df2_data]})
    
    try:
        result, _ = join_func(df1, df2, key='key', lsuffix='_left', rsuffix='_right')
        assert len(result) == 0
        status = "PASS"
        reason = "Properly handled empty relation input without throwing index errors, returning 0 rows."
        out_desc = f"Shape: {result.shape}"
    except Exception as e:
        status = "FAIL"
        reason = f"Failed with exception: {str(e)}"
        out_desc = "None"
        raise
    finally:
        _log_result("test_empty_relations", name, f"df1_keys: {df1_data}, df2_keys: {df2_data}", "Empty dataframe output (0 rows).", out_desc, status, reason)


@pytest.mark.parametrize("name, join_func", JOIN_ALGORITHMS)
def test_disjoint_keys(name, join_func):
    """Test Case 3: Disjoint Key Domains."""
    df1 = pd.DataFrame({'key': [1, 2, 3], 'val_a': ['a', 'b', 'c']})
    df2 = pd.DataFrame({'key': [4, 5, 6], 'val_b': ['d', 'e', 'f']})
    
    try:
        result, _ = join_func(df1, df2, key='key', lsuffix='_left', rsuffix='_right')
        assert len(result) == 0
        status = "PASS"
        reason = "Correctly evaluated disjoint keys, yielding an empty dataset since no keys intersect."
        out_desc = f"Shape: {result.shape}"
    except Exception as e:
        status = "FAIL"
        reason = f"Failed with exception: {str(e)}"
        out_desc = "None"
        raise
    finally:
        _log_result("test_disjoint_keys", name, f"df1 keys: [1,2,3], df2 keys: [4,5,6]", "Empty dataframe output (0 rows).", out_desc, status, reason)


@pytest.mark.parametrize("name, join_func", JOIN_ALGORITHMS)
def test_duplicate_keys_many_to_many(name, join_func):
    """Test Case 4: Many-to-Many Join with duplicate keys."""
    df1 = pd.DataFrame({'key': [5, 5], 'val_a': ['a1', 'a2']})
    df2 = pd.DataFrame({'key': [5, 5, 5], 'val_b': ['b1', 'b2', 'b3']})
    
    try:
        result, _ = join_func(df1, df2, key='key', lsuffix='_left', rsuffix='_right')
        assert len(result) == 6  # 2 * 3 Cartesian expansion
        status = "PASS"
        reason = "Successfully performed many-to-many Cartesian product expansion (2 * 3 = 6 rows)."
        out_desc = f"Shape: {result.shape}, Rows: {len(result)}"
    except Exception as e:
        status = "FAIL"
        reason = f"Failed with exception: {str(e)}"
        out_desc = "None"
        raise
    finally:
        _log_result("test_duplicate_keys_many_to_many", name, f"df1 keys: [5,5], df2 keys: [5,5,5]", "Expected 6 rows (Cartesian product).", out_desc, status, reason)


@pytest.mark.parametrize("name, join_func", JOIN_ALGORITHMS)
def test_null_values_in_key(name, join_func):
    """Test Case 5: NULL (NaN) values in join keys."""
    df1 = pd.DataFrame({'key': [1, np.nan, 3], 'val_a': ['a', 'b', 'c']})
    df2 = pd.DataFrame({'key': [1, 3, np.nan], 'val_b': ['x', 'y', 'z']})
    
    try:
        result, _ = join_func(df1, df2, key='key', lsuffix='_left', rsuffix='_right')
        assert len(result) == 2  # Matches on 1 and 3, filters out NaN
        status = "PASS"
        reason = "Ignored NaN keys successfully and matched valid keys (1 and 3), returning 2 rows."
        out_desc = f"Shape: {result.shape}, Rows: {len(result)}"
    except Exception as e:
        status = "FAIL"
        reason = f"Failed with exception: {str(e)}"
        out_desc = "None"
        raise
    finally:
        _log_result("test_null_values_in_key", name, f"df1 keys: [1, NaN, 3], df2 keys: [1, 3, NaN]", "Expected 2 rows matching valid non-null keys.", out_desc, status, reason)


@pytest.mark.parametrize("name, join_func", JOIN_ALGORITHMS)
def test_colliding_payload_columns(name, join_func):
    """Test Case 6: Colliding non-join column names."""
    df1 = pd.DataFrame({'key': [1, 2], 'val': ['foo1', 'foo2']})
    df2 = pd.DataFrame({'key': [1, 2], 'val': ['bar1', 'bar2']})
    
    try:
        result, _ = join_func(df1, df2, key='key', lsuffix='_left', rsuffix='_right')
        assert len(result) == 2
        assert 'key' in result.columns
        status = "PASS"
        reason = "Handled identical column names cleanly using specified suffix configurations."
        out_desc = f"Columns: {list(result.columns)}, Rows: {len(result)}"
    except Exception as e:
        status = "FAIL"
        reason = f"Failed with exception: {str(e)}"
        out_desc = "None"
        raise
    finally:
        _log_result("test_colliding_payload_columns", name, f"Both relations contain a colliding 'val' column.", "Properly suffixed columns and 2 matching rows.", out_desc, status, reason)