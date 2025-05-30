"""
WARINING: This script is depracated, as it only supports sift1m (not the first 1m of the SIFT1B), 
	and it uses CPU python to search the graph.

This script is used to (a) construct NSG using the C++ binaries, and (b) conduct search.

Example Usage:

python construct_and_search_nsg.py \
    --data_path /mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_base.fvecs \
    --query_path /mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_query.fvecs \
    --gt_path /mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_groundtruth.ivecs \
    --knng_path ./sift_200nn.graph \
    --nsg_path ./sift1m.nsg
"""


import numpy as np
import argparse
import os
import glob

from nsg import IndexNSG, Neighbor
from utils import load_data, ivecs_read

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_base.fvecs", help='Path to the base data file in fvecs format')
    parser.add_argument('--query_path', type=str, default="/mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_query.fvecs", help='Path to the query data file in fvecs format')
    parser.add_argument('--gt_path', type=str, default="/mnt/scratch/wenqi/Faiss_experiments/sift1M/sift_groundtruth.ivecs", help='Path to the groundtruth file in ivecs format')
    parser.add_argument('--knng_path', type=str, default="./sift_200nn.graph", help='Path to the KNNG graph file')
    parser.add_argument('--nsg_path', type=str, default="./sift1m.nsg", help='Path to the NSG index file')

    # TODO 1: Add parameters for the search algorithm
    # NSG construction parameters
    L = 100
    R = 64 # width / M in HNSW
    C = 500
    # Search parameters
    K = 100
    L_search = 100
    assert(L_search >= K)

    args = parser.parse_args()


    '''
    Part 1: Load data and build the NSG index
    '''
    print("Start loading data")
    data_load, points_num, dim = load_data(args.data_path)
    query_load, query_num, query_dim = load_data(args.query_path)
    assert(dim == query_dim)
    gt_load = ivecs_read(args.gt_path)

    print("data_load: ", np.shape(data_load))
    print("query_load: ", np.shape(query_load))
    print("gt_load: ", np.shape(gt_load))

    index = IndexNSG(dim, points_num)

    if glob.glob(args.nsg_path):
        print("NSG index file already exists")
    else:
        nsg_construct_path = "../nsg/build/tests/test_nsg_index"
        if not glob.glob(nsg_construct_path):
            print(f"NSG construction code: {nsg_construct_path} not found")
            exit()
        # assert path exists
        assert os.path.exists(args.data_path)
        assert os.path.exists(args.knng_path)
        
        cmd_build_nsg = f"{nsg_construct_path} {args.data_path} {args.knng_path} {L} {R} {C} {args.nsg_path}"
        print(f"Building NSG by running command:\n{cmd_build_nsg}")
        os.system(cmd_build_nsg)

        if not glob.glob(args.nsg_path):
            print("NSG index file construction failed")
            exit()
        print("NSG index file construction succeeded")


    '''
    Part 2: Perform the search
    '''
    index.Load(args.nsg_path)

    qsize = 100
    paras = {'L_search': L_search}

    total = 0
    correct = 0
    total_counter = 0

    print("Start searching (single-threaded)")
    for i in range(qsize):
        indices, node_counter = index.search_with_base_graph(query_load[i], data_load, K, paras)
        total_counter += node_counter
        gt = gt_load[i]
        print(f"query {i}: gt id: {gt[0]}\tsearch result: {indices[0]}")
        # print(len(gt))
        g = set(gt)
        total += len(gt)
        
        for item in indices:
            if item in g:
                correct += 1

    recall = 1.0 * correct / total
    avg_counter = 1.0 * total_counter / qsize
    print("recall: ", recall, "avg_counter: ", avg_counter)