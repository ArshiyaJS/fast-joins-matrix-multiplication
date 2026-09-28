import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Set

def seek(sorted_array: np.ndarray, value: Any, start_idx: int = 0) -> Tuple[Any, int]:
    """
    Core Leapfrog primitive with safe bounds checking and zero-copy offset search.
    Finds the first element in sorted_array[start_idx:] that is >= value.
    """
    if sorted_array is None or start_idx >= len(sorted_array):
        return None, -1
    
    # Use binary search on the slice view (numpy views do not copy data)
    sub_stream = sorted_array[start_idx:]
    local_idx = np.searchsorted(sub_stream, value, side='left')
    
    if local_idx < len(sub_stream):
        absolute_idx = start_idx + local_idx
        return sorted_array[absolute_idx], absolute_idx
        
    return None, -1


def determine_connected_variable_order(query: List[Dict[str, Any]]) -> List[str]:
    """
    Fixes Trap 1 (Connectedness Problem): 
    Builds a connected variable order starting from the smallest domain variable 
    and iteratively picking next variables that share a relation with already bound variables.
    """
    all_vars = set()
    var_domains: Dict[str, Set[Any]] = {}
    rel_map = [] # list of (rel_name, set_of_attrs)
    
    for rel in query:
        attrs = rel['attrs']
        rel_map.append((rel['name'], set(attrs)))
        for attr in attrs:
            all_vars.add(attr)
            if attr not in var_domains:
                var_domains[attr] = set()
            var_domains[attr].update(rel['table'][attr].unique())

    # Pick the first variable with the absolute smallest domain
    first_var = min(all_vars, key=lambda v: len(var_domains[v]))
    
    bound_vars = {first_var}
    ordered_vars = [first_var]
    
    # Greedily add connected variables
    while len(bound_vars) < len(all_vars):
        candidates = []
        for v in all_vars - bound_vars:
            # Check if variable 'v' shares any relation with currently bound variables
            isConnected = any(v in rel_attrs and not rel_attrs.isdisjoint(bound_vars) for _, rel_attrs in rel_map)
            if isConnected:
                candidates.append(v)
                
        if not candidates:
            # Fallback if graph is disconnected: pick smallest domain among remaining
            remaining = all_vars - bound_vars
            next_var = min(remaining, key=lambda v: len(var_domains[v]))
        else:
            # Pick candidate with smallest domain size
            next_var = min(candidates, key=lambda v: len(var_domains[v]))
            
        bound_vars.add(next_var)
        ordered_vars.append(next_var)
        
    return ordered_vars


class RobustGeneralLFTJEngine:
    def __init__(self, query: List[Dict[str, Any]]):
        self.query = query
        self.variable_order = determine_connected_variable_order(query)
        
        # Pre-index tables using MultiIndexes for high-performance slicing
        self.indexed_relations = {}
        for rel in query:
            name = rel['name']
            df = rel['table']
            attrs = rel['attrs']
            
            relation_indices = {}
            for i in range(1, len(attrs) + 1):
                prefix_attrs = attrs[:i]
                sorted_df = df.sort_values(by=prefix_attrs).reset_index(drop=True)
                
                if len(prefix_attrs) > 1:
                    m_idx = pd.MultiIndex.from_frame(sorted_df[prefix_attrs])
                    relation_indices[tuple(prefix_attrs)] = {
                        'type': 'multi',
                        'index': m_idx,
                        'df': sorted_df,
                        'leaf_attr': prefix_attrs[-1]
                    }
                else:
                    col = prefix_attrs[0]
                    unique_vals = np.sort(sorted_df[col].unique())
                    relation_indices[tuple(prefix_attrs)] = {
                        'type': 'single',
                        'values': unique_vals,
                        'df': sorted_df,
                        'leaf_attr': col
                    }
                    
            self.indexed_relations[name] = {
                'attrs': attrs,
                'indices': relation_indices,
                'raw_df': df
            }

    def _get_candidates_for_variable(self, var: str, partial_assignment: Dict[str, Any]) -> np.ndarray:
        """
        Robust Candidate Retrieval: Uses pre-built MultiIndexes and C-optimized .get_loc() slices
        for fast prefix lookups, avoiding slow sequential DataFrame filtering loops.
        """
        active_streams = []
        
        for rel in self.query:
            name = rel['name']
            rel_info = self.indexed_relations[name]
            attrs = rel_info['attrs']
            
            if var in attrs:
                indices = rel_info['indices']
                
                # Determine the longest contiguous prefix of attrs starting from attrs[0] present in partial_assignment
                k = 0
                while k < len(attrs) and attrs[k] in partial_assignment:
                    k += 1
                
                sub_df = None
                if k > 0:
                    prefix_attrs = tuple(attrs[:k])
                    key = tuple(partial_assignment[attrs[i]] for i in range(k))
                    index_info = indices.get(prefix_attrs)
                    
                    if index_info is not None:
                        sorted_df = index_info['df']
                        if index_info['type'] == 'multi':
                            m_idx = index_info['index']
                            try:
                                loc = m_idx.get_loc(key)
                                sub_df = sorted_df.iloc[loc]
                            except KeyError:
                                sub_df = pd.DataFrame(columns=attrs)
                        else:  # single attribute index
                            vals = index_info['values']
                            val = key[0]
                            idx = np.searchsorted(vals, val, side='left')
                            if idx < len(vals) and vals[idx] == val:
                                start_idx = np.searchsorted(sorted_df[prefix_attrs[0]], val, side='left')
                                end_idx = np.searchsorted(sorted_df[prefix_attrs[0]], val, side='right')
                                sub_df = sorted_df.iloc[start_idx:end_idx]
                            else:
                                sub_df = pd.DataFrame(columns=attrs)
                
                if sub_df is None:
                    sub_df = rel_info['raw_df']
                
                # Filter any remaining bound attributes not covered by the primary prefix
                bound_conditions = {
                    attr: partial_assignment[attr] 
                    for attr in attrs 
                    if attr in partial_assignment and (sub_df is rel_info['raw_df'] or attr not in attrs[:k])
                }
                
                if bound_conditions and not sub_df.empty:
                    for attr, val in bound_conditions.items():
                        sub_df = sub_df[sub_df[attr] == val]
                
                if sub_df.empty:
                    return np.array([])
                
                candidates = np.sort(sub_df[var].unique())
                active_streams.append(candidates)

        if not active_streams:
            return np.array([])

        # Sort streams by length for optimal leapfrog intersection performance
        active_streams.sort(key=lambda x: len(x))
        
        iters = [0] * len(active_streams)
        stream_count = len(active_streams)
        common_candidates = []
        
        # Leapfrog Intersection Core Loop
        while True:
            stream_0 = active_streams[0]
            if iters[0] >= len(stream_0):
                break
            max_val = stream_0[iters[0]]
            
            i = 1
            restart = False
            while i < stream_count:
                stream = active_streams[i]
                curr_idx = iters[i]
                
                # Safe bound-checked seek
                val, absolute_idx = seek(stream, max_val, start_idx=curr_idx)
                if val is None:
                    return np.array(common_candidates) # Stream exhausted
                    
                iters[i] = absolute_idx
                
                if val > max_val:
                    max_val = val
                    restart = True
                    break
                i += 1
                
            if restart:
                # Find new position in stream 0 matching the new max_val
                val_0, absolute_idx_0 = seek(stream_0, max_val, start_idx=iters[0])
                if val_0 is None:
                    return np.array(common_candidates)
                iters[0] = absolute_idx_0
                continue
                
            # All streams match max_val
            common_candidates.append(max_val)
            iters[0] += 1
            
        return np.array(common_candidates)

    def execute(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        results = []

        def recursive_join(var_idx: int, current_assignment: Dict[str, Any]):
            if var_idx == len(self.variable_order):
                # Bag Semantics Multiplicity Calculation:
                # Multiply the occurrence count of this binding tuple across all base relations
                mult = 1
                for rel in self.query:
                    raw_df = rel['table']
                    mask = np.ones(len(raw_df), dtype=bool)
                    for attr, val in current_assignment.items():
                        if attr in raw_df.columns:
                            mask &= (raw_df[attr].to_numpy() == val)
                    count = mask.sum()
                    mult *= count
                    if mult == 0:
                        break
                
                if mult > 0:
                    for _ in range(mult):
                        results.append(current_assignment.copy())
                return

            current_var = self.variable_order[var_idx]
            candidates = self._get_candidates_for_variable(current_var, current_assignment)
            
            for val in candidates:
                current_assignment[current_var] = val
                recursive_join(var_idx + 1, current_assignment)
                del current_assignment[current_var]

        recursive_join(0, {})

        df_out = pd.DataFrame(results)
        diagnostics = {
            "info": "Robust General LFTJ Engine Executed with Bag Semantics",
            "variable_order": self.variable_order,
            "output_rows": len(df_out)
        }
        return df_out, diagnostics  