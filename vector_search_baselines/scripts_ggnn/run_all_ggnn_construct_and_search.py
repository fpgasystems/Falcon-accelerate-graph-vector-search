'''
This script runs ggnn to either (a) construct index or (b) measure the recall and search time.

Example Usage (construct):

    python run_all_ggnn_construct_and_search.py --mode construct \
        --ggnn_index_path ../data/GPU_GGNN_GRAPH/ \
        --ggnn_bin_path ../ggnn/build_local/ \
        --dataset SIFT1M \
        --gpu_id 3

Example Usage (search):

    python run_all_ggnn_construct_and_search.py --mode search \
        --ggnn_index_path ../data/GPU_GGNN_GRAPH/ \
        --ggnn_bin_path ../ggnn/build_local/ \
        --dataset SIFT1M \
        --gpu_id 3 \
        --nruns 3
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


def read_from_log(log_fname:str):
    """
    Read recall and latency_ms_per_batch from the log file
    """
    num_batch = 0
    latency_ms_per_batch = []
    qps = 0.0
    recall_1 = None
    recall_10 = None

    with open(log_fname, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if "query_num divided into" in line:
                num_batch = int(line.split(" ")[7])
                # recorded in ms
                for i in range(num_batch):
                    values = lines[idx + 1 + i].split(" ")
                    latency_ms_per_batch.append(float(values[11]))
            elif "c@1 (=r@1):" in line:
                recall_1 = float(line.split(" ")[6])
            elif "c@10:" in line:
                recall_10 = float(line.split(" ")[5])
            elif "Query_per_second:" in line:
                qps = float(line.split(" ")[5])

    # if more than 2 batches, remove the latency of the first and last batch
    if len(latency_ms_per_batch) > 2:
        latency_ms_per_batch = latency_ms_per_batch[1:-1]	

    return recall_1, recall_10, latency_ms_per_batch, qps


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='search', help='mode to use: either construct or search', choices=['construct', 'search'])
    parser.add_argument('--ggnn_index_path', type=str, default="../data/GPU_GGNN_index", help='path to output constructed ggnn index')
    parser.add_argument('--perf_df_path', type=str, default='perf_df.pickle', help='path to save the performance dataframe')
    parser.add_argument('--ggnn_bin_path', type=str, default=None, help='path to ggnn binary directory')
    parser.add_argument('--dataset', type=str, default='SIFT1M', help='dataset to use: SIFT1M')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu core to use')
    parser.add_argument('--nruns', type=int, default=3, help='number of runs per setting (for recording latency and throughput)')
    
    args = parser.parse_args()
    
    mode = args.mode
    ggnn_index_path = args.ggnn_index_path
    perf_df_path = args.perf_df_path
    ggnn_bin_path = args.ggnn_bin_path
    dataset = args.dataset
    gpu_id = args.gpu_id
    nruns = args.nruns
    
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

    # construct_KBuild_list = [24, 40]         # number of neighbors per point in the graph
    # construct_S_list = [32, 64]              # segment/batch size (needs to be > KBuild-KF)
    # construct_KQuery_list = [10, 20]         # number of neighbors to search for
    # construct_MaxIter_list = [200, 400, 800] # number of iterations for BFiS
    
    # search_KBuild_list = [24, 40]           # should be subset of construct_KBuild_list
    # search_S_list = [32, 64]                # should be subset of construct_S_list
    # search_KQuery_list = [10, 20]           # should be subset of construct_KQuery_list
    # search_MaxIter_list = [200, 400, 800]   # should be subset of construct_MaxIter_list
    # search_tauq_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    # search_bs_list = [1, 2, 4, 8, 16, 10000]
    
    construct_KBuild_list = [32, 64]         # number of neighbors per point in the graph
    construct_S_list = [64]              # segment/batch size (needs to be > KBuild-KF)
    construct_KQuery_list = [10]         # number of neighbors to search for
    construct_MaxIter_list = [400] # number of iterations for BFiS
    
    search_KBuild_list = [32, 64]           # should be subset of construct_KBuild_list
    search_S_list = [64]                # should be subset of construct_S_list
    search_KQuery_list = [10]
    search_MaxIter_list = [1, 32, 64, 100, 200, 400]
    search_tauq_list = [0.5]
    search_bs_list = [1, 2, 4, 8, 16, 32, 10000]
    # search_bs_list = [10000]
    
    # First make sure everything is compiled
    cmakelist_path = os.path.dirname(ggnn_bin_path)
    cmakelist = os.path.join(cmakelist_path, "CMakeLists.txt")
    if mode == 'construct':
        update_cmakelists(cmakelist, 'KBUILD_VALUES', construct_KBuild_list)
        update_cmakelists(cmakelist, 'SEG_VALUES', construct_S_list)
        update_cmakelists(cmakelist, 'KQUERY_VALUES', construct_KQuery_list)
        update_cmakelists(cmakelist, 'MAXITER_VALUES', construct_MaxIter_list)
    elif mode == 'search':
        update_cmakelists(cmakelist, 'KBUILD_VALUES', search_KBuild_list)
        update_cmakelists(cmakelist, 'SEG_VALUES', search_S_list)
        update_cmakelists(cmakelist, 'KQUERY_VALUES', search_KQuery_list)
        update_cmakelists(cmakelist, 'MAXITER_VALUES', search_MaxIter_list)
    print(f"Compiling GGNN binary...")
    subprocess.run(["cmake", ".."], cwd=ggnn_bin_path, check=True)
    subprocess.run(["make"], cwd=ggnn_bin_path, check=True)
    
    # Now run the experiments
    cmd_core = f"export CUDA_VISIBLE_DEVICES={gpu_id}; "
    print(f"Setting CUDA_VISIBLE_DEVICES: {cmd_core}")
    os.system(cmd_core)
    
    if mode == 'construct':
        
        int_mode = 0
        for KBuild in construct_KBuild_list:
            for S in construct_S_list:
                
                KQuery = construct_KQuery_list[0]
                MaxIter = construct_MaxIter_list[0]
                
                if not os.path.exists(ggnn_index_path):
                    os.mkdir(ggnn_index_path)
                ggnn_index = os.path.join(ggnn_index_path, f"{dataset}_KB{KBuild}_S{S}.bin")
                if os.path.exists(ggnn_index):
                    print(f"GGNN index already exists: {ggnn_index}")
                    continue
                else:
                    ggnn_bin = os.path.join(ggnn_bin_path, f"{ggnn_bin_prefix}_KB{KBuild}_S{S}_KQ{KQuery}_MI{MaxIter}")
                    cmd_construct = cmd_core + f"{ggnn_bin} --dbname={dataset} --graph_filename={ggnn_index} --mode={int_mode} --log_dir=."
                    print(f"Constructing GGNN index: {cmd_construct}")
                    os.system(cmd_construct)

                    if not os.path.exists(ggnn_index):
                        raise ValueError(f"GGNN index construction failed: {ggnn_index}")
                    print(f"GGNN index construction succeeded: {ggnn_index}")
    
    
    elif mode == 'search':
        
        key_columns = ['dataset', 'KBuild', 'S', 'KQuery', 'MaxIter', 'batch_size', 'tau_query']
        result_columns = ['recall_1', 'recall_10', 'latency_ms_per_batch', 'qps']
        columns = key_columns + result_columns
        
        if os.path.exists(perf_df_path): # load existing
            df = pd.read_pickle(perf_df_path)
            assert len(df.columns.values) == len(columns)
            for col in columns:
                assert col in df.columns.values
            print("Performance dataframe loaded")
        else:
            print(f"Performance dataframe does not exist, create a new one: {perf_df_path}")
            df = pd.DataFrame(columns=columns)
        pd.set_option('display.expand_frame_repr', False)
        
        print("Warning: please set the search parameters in the script, current settings:")
        print(f"search_KBuild_list: {search_KBuild_list}")
        print(f"search_S_list: {search_S_list}")
        print(f"search_KQuery_list: {search_KQuery_list}")
        print(f"search_MaxIter_list: {search_MaxIter_list}")
        
        int_mode = 1
        for KBuild in search_KBuild_list:
            for S in search_S_list:
                ggnn_index = os.path.join(ggnn_index_path, f"{dataset}_KB{KBuild}_S{S}.bin")
                if not os.path.exists(ggnn_index):
                    raise ValueError(f"GGNN index does not exist: {ggnn_index}")
                
                for KQuery in search_KQuery_list:
                    for MaxIter in search_MaxIter_list:
                        for bs in search_bs_list:
                            for tauq in search_tauq_list:
                        
                                latency_ms_per_batch = []
                                qps = []
                                for run in range(nruns):
                                    files_before = set(glob.glob("*"))
                                    ggnn_bin = os.path.join(ggnn_bin_path, f"{ggnn_bin_prefix}_KB{KBuild}_S{S}_KQ{KQuery}_MI{MaxIter}")
                                    if not os.path.exists(ggnn_bin):
                                        raise ValueError(f"GGNN binary does not exist: {ggnn_bin}")
                                    cmd_search = cmd_core + f"{ggnn_bin} " + \
                                                f"--dbname={dataset} " + \
                                                f"--graph_filename={ggnn_index} " + \
                                                f"--mode={int_mode} " + \
                                                f"--bs={bs} " + \
                                                f"--tau_query={tauq} " + \
                                                f"--log_dir=."
                                    print(f"Searching GGNN by running command:\n{cmd_search}")
                                    os.system(cmd_search)
                                    
                                    files_after = set(glob.glob("*"))
                                    log_file = list(files_after - files_before)[0]
                                    recall_1, recall_10, latency_ms_per_batch_this_run, qps_this_run = read_from_log(log_file)
                                    latency_ms_per_batch.extend(latency_ms_per_batch_this_run)
                                    qps.append(qps_this_run)
                                    
                                    os.system(f"rm {log_file}")
                                    # print(qps)
                                    
                                # if already in the df, delete the old row first
                                key_values = {
                                    'dataset': dataset,
                                    'KBuild': KBuild,
                                    'S': S,
                                    'KQuery': KQuery,
                                    'MaxIter': MaxIter,
                                    'batch_size': bs,
                                    'tau_query': tauq
                                }
                                
                                idx = df.index[(df['dataset'] == dataset) & \
                                            (df['KBuild'] == KBuild) & \
                                            (df['S'] == S) & \
                                            (df['KQuery'] == KQuery) & \
                                            (df['MaxIter'] == MaxIter) & \
                                            (df['batch_size'] == bs) & \
                                            (df['tau_query'] == tauq)]
                                if len(idx) > 0:
                                    print("Warning: duplicate entry found, deleting the old entry:")
                                    print(df.loc[idx])
                                    df = df.drop(idx)
                                print(f"Appending new entry:")
                                new_entry = {**key_values, 'recall_1': recall_1, 'recall_10': recall_10, 'latency_ms_per_batch': latency_ms_per_batch, 'qps': qps}
                                print(new_entry)
                                df = df.append(new_entry, ignore_index=True)
                                        
                    df.to_pickle(perf_df_path, protocol=4)

                        
                        
                        
                        