'''
This script runs ggnn search infinitely to measure the energy consumption.
[Please wait until the "Start infinte search..." is shown]

Example Usage:

    python run_all_ggnn_inf_search.py \
        --ggnn_index_path ../data/GPU_GGNN_GRAPH/ \
        --ggnn_bin_path ../ggnn/build_local/ \
        --dataset SIFT1M \
        --gpu_id 3 \
        --KBuild 64 \
        --S 64 \
        --KQuery 10 \
        --MaxIter 400 \
        --tauq 0.5 \
        --bs 10000
        
'''

import os
import re
import subprocess
import argparse
import glob
import numpy as np
import pandas as pd
import pickle

def update_cmakelists(file_path, parameter, values:list[int]):
    with open(file_path, 'r') as file:
        content = file.read()
    
    new_line = f"set({parameter} {' '.join(map(str, values))})"
    new_content = re.sub(rf"set\({parameter}.*\)", new_line, content)
    # write back to file
    with open(file_path, 'w') as file:
        file.write(new_content)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ggnn_index_path', type=str, default="../data/GPU_GGNN_index", help='path to output constructed ggnn index')
    parser.add_argument('--ggnn_bin_path', type=str, default=None, help='path to ggnn binary directory')
    parser.add_argument('--dataset', type=str, default='SIFT1M', help='dataset to use: SIFT1M')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu core to use')
    
    parser.add_argument('--KBuild', type=int, default=64, help='number of neighbors to build the graph')
    parser.add_argument('--S', type=int, default=64, help='number of segments')
    parser.add_argument('--KQuery', type=int, default=10, help='number of neighbors to query')
    parser.add_argument('--MaxIter', type=int, default=400, help='max number of iterations')
    parser.add_argument('--tauq', type=float, default=0.5, help='tau query')
    parser.add_argument('--bs', type=int, default=10000, help='batch size')
    
    args = parser.parse_args()
    
    ggnn_index_path = args.ggnn_index_path
    ggnn_bin_path = args.ggnn_bin_path
    dataset = args.dataset
    gpu_id = args.gpu_id
    
    KBuild = args.KBuild
    S = args.S
    KQuery = args.KQuery
    MaxIter = args.MaxIter
    tauq = args.tauq
    bs = args.bs
    
    # convert to absolute path
    ggnn_index_path = os.path.abspath(ggnn_index_path)
    ggnn_bin_path = os.path.abspath(ggnn_bin_path)
    
    if not os.path.exists(ggnn_bin_path):
        raise ValueError(f"GGNN binary path does not exist: {ggnn_bin_path}")
    
    ggnn_bin_prefix = ""
    if "SIFT" in dataset:
        ggnn_bin_prefix = "sift"
    elif "Deep" in dataset:
        ggnn_bin_prefix = "deep"
    elif "SPACEV" in dataset:
        ggnn_bin_prefix = "spacev"
    
    # First make sure everything is compiled
    cmakelist_path = os.path.dirname(ggnn_bin_path)
    cmakelist = os.path.join(cmakelist_path, "CMakeLists.txt")
    update_cmakelists(cmakelist, 'KBUILD_VALUES', [KBuild])
    update_cmakelists(cmakelist, 'SEG_VALUES', [S])
    update_cmakelists(cmakelist, 'KQUERY_VALUES', [KQuery])
    update_cmakelists(cmakelist, 'MAXITER_VALUES', [MaxIter])
    print(f"Compiling GGNN binary...")
    subprocess.run(["cmake", ".."], cwd=ggnn_bin_path, check=True)
    subprocess.run(["make"], cwd=ggnn_bin_path, check=True)
    
    # Now run the experiments
    cmd_core = f"export CUDA_VISIBLE_DEVICES={gpu_id}; "
    print(f"Setting CUDA_VISIBLE_DEVICES: {cmd_core}")
    os.system(cmd_core)

    
    print("Warning: please set the search parameters in the script, current settings:")
    print(f"search_KBuild: {KBuild}")
    print(f"search_S: {S}")
    print(f"search_KQuery: {KQuery}")
    print(f"search_MaxIter: {MaxIter}")
    
    int_mode = 1
    ggnn_index = os.path.join(ggnn_index_path, f"{dataset}_KB{KBuild}_S{S}.bin")
    if not os.path.exists(ggnn_index):
        raise ValueError(f"GGNN index does not exist: {ggnn_index}")
                    
    files_before = set(glob.glob("*"))
    ggnn_bin = os.path.join(ggnn_bin_path, f"{ggnn_bin_prefix}_KB{KBuild}_S{S}_KQ{KQuery}_MI{MaxIter}")
    if not os.path.exists(ggnn_bin):
        raise ValueError(f"GGNN binary does not exist: {ggnn_bin}")
    
    cmd_search = f"{ggnn_bin} " + \
                f"--dbname={dataset} " + \
                f"--graph_filename={ggnn_index} " + \
                f"--mode={int_mode} " + \
                f"--bs={bs} " + \
                f"--inf_search=1 " + \
                f"--tau_query={tauq} " + \
                f"--log_dir=."
    print(f"Searching GGNN by running command:\n{cmd_search}")
    os.system(cmd_search)

                        
                        
                        
                        