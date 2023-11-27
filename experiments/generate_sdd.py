from typing import List, Any
from pysdd.sdd import SddManager, Vtree, WmcManager, SddNode
import numpy as np
from copy import copy
import os
from itertools import groupby
from functools import reduce

class LinearConstraintToSddApproximator:

    def __init__(self, true_sink, false_sink, decimals=2) -> None:
        self.true_sink = true_sink
        self.false_sink = false_sink

    def sdd_or(self, v1, v2):
        return v1 | v2

    def sdd_and(self, v1, v2):
        return v1 & v2
    
    def sdd_not(self, node):
        return ~node

    def to_positive_weights(self, weights: List[float], var_nodes: List[Any], threshold:float):
        weights = copy(weights)
        var_nodes = copy(var_nodes)
        for i in range(len(weights)):
            if weights[i] < 0:
                weights[i] = abs(weights[i])
                threshold += weights[i]
                var_nodes[i] = self.sdd_not(var_nodes[i])
        return weights, var_nodes, threshold


    def get_sdd_gte(self, weights: List[float], var_nodes: List[Any], threshold:float):
        weights, var_nodes, threshold = self.to_positive_weights(weights, var_nodes, threshold)
        sorted_weights_and_nodes = sorted(list(zip(weights, var_nodes)), key=lambda x:-x[0])
        sorted_w = np.array([v[0] for v in sorted_weights_and_nodes])
        weights_remaining_sum = np.sum(sorted_w) - np.cumsum(sorted_w)
        final_node = self.false_sink
        last_layer = [(0, self.true_sink)]
        for (w, node), remaining_w in zip(sorted_weights_and_nodes, weights_remaining_sum):
            new_layer = []
            print(f"\n{remaining_w}", "Size:", len(last_layer))
            # print(last_layer)
            for node_i, (lw, last_node) in enumerate(last_layer):
                print(remaining_w, "Size:", node_i+1, "/", len(last_layer), end="\t\t\t\r")
                high = lw+w
                high_node = self.sdd_and(node, last_node)
                low_node = self.sdd_and(self.sdd_not(node), last_node)
                low = lw
                if high >= threshold:
                    final_node = self.sdd_or(high_node, final_node)
                else:
                    new_layer.append((high, high_node))

                if low + remaining_w < threshold:
                    final_node = self.sdd_and(~low_node, final_node)
                else:
                    new_layer.append((low, low_node))

            if len(new_layer) == 0:
                break
            new_layer = list(set(new_layer))
            simplified_new_layer = []
            key_func = lambda x: np.round(x[0])
            new_layer = sorted(new_layer, key=key_func)
            for key, group in groupby(new_layer, key=key_func):
                group_node = self.false_sink
                for w, node in group:
                    group_node = group_node | node
                simplified_new_layer.append((key, group_node))
            last_layer = simplified_new_layer

        return final_node

        
def get_all_models(sdd_manager, sdd):
    all_models = [[model[i+1] for i in range(sdd_manager.var_count())] for model in list(sdd.models())]
    return np.array(all_models).astype(bool)


def compile_bss_constraint(var_count):
    bits_per_var = 6
    binary_var_count = var_count * bits_per_var
    var_order = np.arange(binary_var_count)+1
    global_sum = 30*var_count
    local_bound = 35
    print("Global sum:", global_sum)
    print("Var order", var_order)
    bit_weight_pattern = 2**(bits_per_var - np.arange(bits_per_var)-1)
    vtree = Vtree(
        var_count=binary_var_count,
        var_order=var_order,
        vtree_type="balanced")
    sddmanager = SddManager.from_vtree(vtree)
    sdd_vars = list(sddmanager.vars)
    print(sdd_vars)
    approximator = LinearConstraintToSddApproximator(sddmanager.true(), sddmanager.false())
    
    assert len(sdd_vars) == binary_var_count

    all_constraints = []
    for i in range(var_count):
        #Constraint:  v >= 1
        bit_weights = np.zeros((var_count, bits_per_var))
        bit_weights[i] = bit_weight_pattern
        bit_weights = bit_weights.reshape(-1)
        print(bit_weights)
        lower_const = approximator.get_sdd_gte(bit_weights, sdd_vars, 0)
        all_constraints.append(lower_const)
        
        #Constraint: v <= 35 ---> -v >= -35 
        upper_count = approximator.get_sdd_gte(-bit_weights, sdd_vars, -local_bound)
        all_constraints.append(upper_count)
    

    # Global constraint
    bit_weights = np.zeros((var_count, bits_per_var))
    bit_weights[:] = bit_weight_pattern
    bit_weights = bit_weights.reshape(-1)
    globale_lower_c = approximator.get_sdd_gte(bit_weights, sdd_vars, global_sum)
    globale_upper_c = approximator.get_sdd_gte(-bit_weights, sdd_vars, -global_sum)
    all_constraints.append(globale_lower_c)
    all_constraints.append(globale_upper_c)
    sdd = reduce(lambda x, y: x&y, all_constraints, sddmanager.true())
    print(sdd)
    print("Sdd node count:", sdd.count())
    print("saving sdd and vtree ... ")
    base_folder = "./logs/sample_generation/sdd"
    os.makedirs(base_folder, exist_ok=True)
    with open(f"{base_folder}/sdd_{var_count}.dot", "w") as out:
        print(sdd.dot(), file=out)
    with open(f"{base_folder}/vtree_{var_count}.dot", "w") as out:
        print(vtree.dot(), file=out)
    

    sdd.save(f"{base_folder}/sdd_file_{var_count}.sdd".encode('utf-8'))
    vtree.save(f"{base_folder}/vtree_file_{var_count}.vtree".encode('utf-8'))

    print("Getting all models")
    all_models = get_all_models(sddmanager, sdd)
    all_values = np.sum((all_models.astype(int) * bit_weights).reshape((-1, var_count, bits_per_var)), axis=2)
    np.save(f"{base_folder}/data_{var_count}.npy", all_values)
    np.save(f"{base_folder}/bits_{var_count}.npy", all_models)
    print("Done")

if __name__ == "__main__":
    compile_bss_constraint(3)
    compile_bss_constraint(5)