'''
Sample command to run the script:
    python run_all_hnsw_inf_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_inf_search_bin_path ../hnswlib/build/inf_search \
     --dataset SIFT1M --max_cores 16 --max_degree 64 --ef 64 --omp 1 --batch_size 10000
'''


import os
import argparse
import json
import glob
import pandas as pd
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--hnsw_inf_search_bin_path', type=str, default=None, help='path to algorithm bin')
    parser.add_argument('--hnsw_index_path', type=str, default="../data/CPU_hnsw_indexes", help='path to output constructed nsg index')
    parser.add_argument('--dataset', type=str, default='SIFT1M', help='dataset to use: SIFT1M')
    parser.add_argument('--max_cores', type=int, default=16, help='maximum number of cores used for search')
    
    parser.add_argument('--max_degree', type=int, default=64, help='max degree')
    parser.add_argument('--ef', type=int, default=64, help='ef')
    parser.add_argument('--omp', type=int, default=1, help='enable omp')
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size')

    args = parser.parse_args()

    hnsw_inf_search_bin_path:str = args.hnsw_inf_search_bin_path
    hnsw_index_path:str = args.hnsw_index_path
    dataset:str = args.dataset
    max_cores:int = args.max_cores
    
    max_degree = args.max_degree
    ef = args.ef
    omp = args.omp
    batch_size = args.batch_size

    hnsw_index_path = os.path.abspath(hnsw_index_path)

    # random num log
    log_perf_test = 'hnsw_{}.log'.format(np.random.randint(1000000))

    if not os.path.exists(hnsw_inf_search_bin_path):
        raise ValueError(f"Path to algorithm does not exist: {hnsw_inf_search_bin_path}")


    hnsw_index_file = os.path.join(hnsw_index_path, f"{dataset}_index_MD{max_degree}.bin")
    assert os.path.exists(hnsw_index_file)

    if omp == 0:
        interq_multithread = 1
    else:
        if batch_size > max_cores:
            interq_multithread = max_cores
        else:
            interq_multithread = batch_size
    

    cmd_search_hnsw = f"taskset --cpu-list 0-{max_cores-1} " + \
                        f"{hnsw_inf_search_bin_path} " + \
                        f"{dataset} " + \
                        f"{hnsw_index_file} " + \
                        f"{ef} " + \
                        f"{omp} " + \
                        f"{interq_multithread} " + \
                        f"{batch_size} "
    print(f"Searching HNSW by running command:\n{cmd_search_hnsw}")
    os.system(cmd_search_hnsw)
        