'''
This script runs c++ nsg search infinitely to measure the energy consumption.

Example Usage:

python run_all_nsg_inf_search.py \
    --nsg_inf_search_bin_path ../nsg/build/tests/test_nsg_optimized_inf_search \
    --dataset SIFT1M --max_cores 16 \
    --construct_R 64 --search_L 64 --omp 1 --batch_size 10000
    
    The latest result can be viewed in temp log: "./nsg.log".
    Currently we support SIFT, Deep, and SBERT1M.
'''
import os
import argparse
import glob
import numpy as np
import pandas as pd


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_nsg_path', type=str, default="../data/CPU_NSG_index", help='path to output constructed nsg index')
    parser.add_argument('--nsg_inf_search_bin_path', type=str, default=None, help='path to nsg infinite search algorithm')
    parser.add_argument('--dataset', type=str, default='SIFT1M', help='dataset to use: SIFT1M')
    parser.add_argument('--max_cores', type=int, default=16, help='maximum number of cores used for search')
    
    parser.add_argument('--construct_R', type=int, default=64, help='max degree')
    parser.add_argument('--search_L', type=int, default=64, help='search L')
    parser.add_argument('--omp', type=int, default=1, help='enable omp')
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size')
    

    args = parser.parse_args()

    nsg_inf_search_bin_path:str = args.nsg_inf_search_bin_path
    output_nsg_path = args.output_nsg_path
    dataset:str = args.dataset
    max_cores = args.max_cores
    
    construct_R = args.construct_R
    search_L = args.search_L
    omp = args.omp
    batch_size = args.batch_size

    # convert to absolute path
    output_nsg_path = os.path.abspath(output_nsg_path)
    
    # random num log
    # log_nsg = 'nsg_{}.log'.format(np.random.randint(1000000))
            

    if not os.path.exists(nsg_inf_search_bin_path):
        raise ValueError(f"Path to algorithm does not exist: {nsg_inf_search_bin_path}")

    print("Warning: please set the search parameters in the script, current settings:")
    print(f"construct_R_list: {construct_R}")
    print(f"search_L_list: {search_L}")	
    print(f"omp_list: {omp}")
    print(f"batch_size_list: {batch_size}")

    nsg_file = os.path.join(output_nsg_path, f"{dataset}_index_MD{construct_R}.nsg")
    assert os.path.exists(nsg_file)

    # Perform the search
    if omp == 0:
        interq_multithread = 1
    else:
        if batch_size > max_cores:
            interq_multithread = max_cores
        else:
            interq_multithread = batch_size
    
    cmd_search_nsg = f"taskset --cpu-list 0-{max_cores-1} " + \
                    f"{nsg_inf_search_bin_path} " + \
                    f"{dataset} " + \
                    f"{nsg_file} " + \
                    f"{search_L} " + \
                    f"{omp} " + \
                    f"{interq_multithread} " + \
                    f"{batch_size}"
    print(f"Searching NSG by running command:\n{cmd_search_nsg}")
    os.system(cmd_search_nsg)
