"""
To compare the search performance of the full graph approach and the sub-graph approach, we need to construct:
    
    1. The full graph index (e.g. SIFT1M_index_MD64_par1_0.bin)
    
    2. The sub-graph indexes (e.g. SIFT1M_index_MD64_par4_x.bin)
        
        The first two must share the same efConstruction, efSearch, and M parameters.
        Both of them are used for the c++ search and get_items() operation in conversion.

Example Usage:
    python subgraph_vs_full_graph_hnsw.py --dbname SIFT1M --ef_construction 128 --MD 64 \
        --hnsw_path ../data/CPU_hnsw_indexes --subgraph_result_path ../data/sub_graph_results
"""
import argparse
import sys
import os
import time 
import pandas as pd

import hnswlib
import numpy as np

from utils import mmap_fvecs, mmap_bvecs, ivecs_read, fvecs_read, mmap_bvecs_SBERT, \
    read_deep_ibin, read_deep_fbin, print_recall, calculate_recall, read_spacev_int8bin


def read_output_file(filename):
    with open(filename, 'r') as file:
        I = []
        for line in file:
            elements = line.split()
            row = []
            for i in range(0, len(elements), 2):
                first = float(elements[i])
                second = float(elements[i + 1])
                row.append((first, second))
            I.append(row)
    return I


def convert_ids_to_full_graph(I:list[list], N_offset):
    updated_I = []
    for row in I:
        updated_row = [(x[0], x[1] + N_offset) for x in row]
        updated_I.append(updated_row)
    return updated_I


def sort_subgraph_results(I:list[list[list]]):
    sorted_I = []
    for qid in range(len(I[0])):
        sorted_res = []
        for I_sub in I:
            sorted_res.extend(I_sub[qid])
            sorted_res.sort(key=lambda x: x[0])
        sorted_I.append([x[1] for x in sorted_res])
    return sorted_I

def read_from_log(log_fname:str, qnum=10000):
    """
    Read recall and node_count from the log file
    """
    node_counter = 0
    recall_1 = None
    recall_10 = None

    node_counter_per_query = [[] for _ in range(qnum)]
    with open(log_fname, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if "recall_1:" in line:
                recall_1 = float(line.split(" ")[1])
            elif "recall_10:" in line:
                recall_10 = float(line.split(" ")[1])
            elif "total node counter per query:" in line:
                node_counter = int(line.split(" ")[5])
            elif "per query node count" in line:
                # identify query id
                # std::cout << "per query node count " << i <<  ": " ;
                elements = line.replace("per query node count ", "").replace(":", "").split(" ") # ['0', '763', '713', '\n']
                # print(elements)
                qid = int(elements[0])
                # print(qid)
                node_counter_this_query = [int(e) for e in elements[1:-1]]
                node_counter_per_query[qid] = node_counter_this_query

    return recall_1, recall_10, node_counter, node_counter_per_query


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--perf_df_path', type=str, default='perf_subgraph_df.pickle')
    parser.add_argument('--dbname', type=str, default="SIFT1M", help='name of the database, e.g., SIFT10M, Deep10M, GLOVE')
    parser.add_argument('--ef_construction', type=int, default=128, help='ef construction parameter')
    parser.add_argument('--MD', type=int, default=64, help='Max degree of base layer, M * 2 === M0 == MD')
    # parser.add_argument('--ef_full', type=int, default=64, help='ef search parameter on full graph')
    # parser.add_argument('--ef_sub', type=int, default=64, help='ef search parameter on each sub-graph')
    parser.add_argument('--hnsw_path', type=str, default="../data/CPU_hnsw_indexes", help='Path to the HNSW index file')
    parser.add_argument('--subgraph_result_path', type=str, default="", help='Path to the subgraph search result')
    # parser.add_argument('--sub_graph_num', type=int, default=1, help='Number of partitions')
    args = parser.parse_args()
    
    perf_df_path:str = args.perf_df_path
    dbname = args.dbname
    ef_construction = args.ef_construction
    # ef_full = args.ef_full
    # ef_sub = args.ef_sub
    hnsw_path = args.hnsw_path
    subgraph_result_path = args.subgraph_result_path
    # sub_graph_num = args.sub_graph_num
    M = int(args.MD / 2)
    print("MD: {} Derived M in HNSW: {}".format(args.MD, M))
    
    if not os.path.exists(subgraph_result_path):
        os.makedirs(subgraph_result_path)
    log_perf_test = 'hnsw_{}.log'.format(np.random.randint(1000000))

    if dbname.startswith('SIFT'):
        # SIFT1M to SIFT1000M
        dbsize = int(dbname[4:-1])
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/bigann'
        xb = mmap_bvecs(os.path.join(dataset_dir, 'bigann_base.bvecs'))
        xq = mmap_bvecs(os.path.join(dataset_dir, 'bigann_query.bvecs'))
        gt = ivecs_read(os.path.join(dataset_dir, 'gnd/idx_%dM.ivecs' % dbsize))

        # trim xb to correct size
        xb = xb[:dbsize * 1000 * 1000]
        # print(xb[0])
        # print(xb[250000])
        

        # Wenqi: load xq to main memory and reshape
        xq = xq.astype('float32').copy()
        xq = np.array(xq, dtype=np.float32)
        gt = np.array(gt, dtype=np.int32)

    elif dbname.startswith('Deep'):
        # Deep1M to Deep1000M
        dbsize = int(dbname[4:-1])
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/deep1b'

        xb = read_deep_fbin(os.path.join(dataset_dir, 'base.1B.fbin'))
        xq = read_deep_fbin(os.path.join(dataset_dir, 'query.public.10K.fbin'))
        # trim xb to correct size
        xb = xb[:dbsize * 1000 * 1000]
        gt = read_deep_ibin(os.path.join(dataset_dir, 'gt_idx_%dM.ibin' % dbsize))

    elif dbname.startswith('GLOVE'):
        dbsize = 2
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/GLOVE_840B_300d'
    
        xb = read_deep_fbin(os.path.join(dataset_dir, 'glove.840B.300d.fbin'))
        xq = read_deep_fbin(os.path.join(dataset_dir, 'query_10K.fbin'))
        gt = read_deep_ibin(os.path.join(dataset_dir, 'gt_idx_%dM.ibin' % dbsize))

    elif dbname.startswith('SBERT1M'):
        dbsize = 1
        # assert dbname[:5] == 'SBERT' 
        # assert dbname[-1] == 'M'
        # dbsize = int(dbname[5:-1]) # in million
        
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/sbert'
        xb = mmap_bvecs_SBERT(os.path.join(dataset_dir, 'sbert1M.fvecs'), num_vec=int(dbsize * 1e6))
        # xb = mmap_bvecs_SBERT(os.path.join(dataset_dir, 'sbert3B.fvecs'), num_vec=int(dbsize * 1e6))
        xq = mmap_bvecs_SBERT(os.path.join(dataset_dir, 'query_10K.fvecs'), num_vec=10 * 1000)
        gt = read_deep_ibin(os.path.join(dataset_dir, 'gt_idx_%dM.ibin' % dbsize))

        # trim to correct size
        xb = xb[:dbsize * 1000 * 1000]
    
    elif dbname.startswith('SPACEV1M'):
        dbsize = 1
        dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/SPACEV'
        xb = read_spacev_int8bin(os.path.join(dataset_dir, 'vectors_all.bin'))
        xq = read_spacev_int8bin(os.path.join(dataset_dir, 'query_10K.bin'))
        gt = read_deep_ibin(os.path.join(dataset_dir, 'gt_idx_%dM.ibin' % dbsize))
        
        xb = xb[:dbsize * 1000 * 1000]
        
    else:
        print('unknown dataset', dbname, file=sys.stderr)
        sys.exit(1)

    print("Vector shapes:")
    print("Base vector xb: ", xb.shape)
    print("Query vector xq: ", xq.shape)
    print("Ground truth gt: ", gt.shape)

    key_columns = ['dataset', 'max_degree', 'mode', 'sub_graph_num','ef']
    results_columns = ['recall_1', 'recall_10', 'node_counter', 'node_counter_per_query']
    columns = key_columns + results_columns

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

    
    ef_list = [4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192]  
    sub_graph_num_list = [2, 4, 8, 16]
      
    for ef in ef_list:
        for sub_graph_num in sub_graph_num_list:
        
            print(f"Start search with ef = {ef}, sub_graph_num = {sub_graph_num}")
            # Construct the full graph index
            index_path_full = os.path.join(hnsw_path, '{}_index_MD{}_par1_0.bin'.format(dbname, args.MD))
            if not os.path.exists(index_path_full):
                # Load the entire graph and build full index
                print(f"Start full index construction for [{dbname}]")
                p_full = hnswlib.Index(space='l2', dim=xb.shape[1])  # possible options are l2, cosine or ip
                p_full.init_index(max_elements=xb.shape[0], ef_construction=ef_construction, M=M)
                p_full.add_items(xb)
                p_full.save_index(index_path_full)
            else:
                print(f"Index file [{index_path_full}] already exists.")
                # # Load the index
                # print(f"Loading full index for [{dbname}]")
                # p_full = hnswlib.Index(space='l2', dim=xb.shape[1])
                # p_full.load_index(index_path_full)
                
            # Construct the sub-graph indexes
            p = []
            for i in range(sub_graph_num):

                N_VEC = int(dbsize * 1000 * 1000 / sub_graph_num)
                xb_sub = xb[i * N_VEC : (i+1) * N_VEC]
                # print(xb_sub[0])
                dim = xb_sub.shape[1] # should be 128
                nq = xq.shape[0]

                index_path_sub = os.path.join(hnsw_path, '{}_index_MD{}_par{}_{}.bin'.format(dbname, args.MD, sub_graph_num, i))
                
                # train index if not exist
                if not os.path.exists(index_path_sub):
                    print(f"Start subgraph index construction for [{dbname}], subgraph [{i}]")
                    # Declaring index
                    p.append(hnswlib.Index(space='l2', dim=dim))  # possible options are l2, cosine or ip
                    p[i].init_index(max_elements=N_VEC, ef_construction=ef_construction, M=M)

                    batch_size = 10000 # add to graph
                    batch_num = int(np.ceil(N_VEC / batch_size))
                    for j in range(batch_num):
                        # print("Adding {} th batch of {} elements".format(i, batch_size))
                        xbatch = xb_sub[j * batch_size: (j + 1) * batch_size]
                        p[i].add_items(xbatch)
                    
                    # Serializing and deleting the index:
                    print("Saving index to '%s'" % index_path_sub)
                    p[i].save_index(index_path_sub)
                
                else:
                    print(f"Index file [{index_path_sub}] already exists.")
                    # # Load the index
                    # print(f"Loading subgraph index for [{dbname}], subgraph [{i}]")
                    # p.append(hnswlib.Index(space='l2', dim=dim))
                    # p[i].load_index(index_path_sub)


            batch_size = 1 # to show per query node count, need to use batch size of 1
            # Run c++ binary for subgraph search
            # SIFT1M /mnt/scratch/hanghu/CPU_hnsw_index/SIFT1M_index_MD64_par4 32 0 1 10000
            hnsw_index_prefix = os.path.join(hnsw_path, '{}_index_MD{}_par{}'.format(dbname, args.MD, sub_graph_num))
            # There are some fixed search parameters
            cmd_hnsw_search = f"../hnswlib/build/subgraph {dbname} {hnsw_index_prefix} {sub_graph_num} {ef} 0 1 {batch_size} {subgraph_result_path} > {log_perf_test}"
            print(f"Running sub-graph search command: {cmd_hnsw_search}")
            os.system(cmd_hnsw_search)
            _, _, node_counter_sub, node_counter_per_query_sub = read_from_log(log_perf_test)
            
            # Run c++ binary for full graph search: sub_graph_num = 1
            hnsw_index_prefix = os.path.join(hnsw_path, '{}_index_MD{}_par1'.format(dbname, args.MD))
            cmd_hnsw_search = f"../hnswlib/build/subgraph {dbname} {hnsw_index_prefix} 1 {ef} 0 1 {batch_size} {subgraph_result_path} > {log_perf_test}"
            print(f"Running full-graph search command: {cmd_hnsw_search}")
            os.system(cmd_hnsw_search)
            recall_1_full, recall_10_full, node_counter_full, node_counter_per_query_full = read_from_log(log_perf_test)
            
            # os.system(f"rm {log_perf_test}")

            # Convert sub-graph result ids to full graph ids
            # and calculate recall
            recall_1_sub = 0
            recall_10_sub = 0
            I = []
            for i in range(sub_graph_num):
                # read search result
                subgraph_result_file = os.path.join(subgraph_result_path, f'{dbname}_{sub_graph_num}_{i}.txt')
                I_sub = read_output_file(subgraph_result_file)
                I_sub = convert_ids_to_full_graph(I_sub, i * N_VEC)
                I.append(I_sub)
                
                # print(I_sub[0])
            
            # Merge and sort the sub-graph search results
            sorted_I = sort_subgraph_results(I)
            sorted_I = np.array(sorted_I)
            
            recall_1_sub = calculate_recall(sorted_I, gt, 1)
            recall_10_sub = calculate_recall(sorted_I, gt, 10)
        
            # Save the full graph results to the dataframe
            key_values = {
                'dataset': dbname,
                'max_degree': args.MD,
                'mode': 'full',
                'sub_graph_num': '1',
                'ef': ef
            }
            
            idx = df.index[(df['dataset'] == dbname) & \
                            (df['max_degree'] == args.MD) & \
                            (df['mode'] == 'full') & \
                            (df['sub_graph_num'] == '1') & \
                            (df['ef'] == ef)]
            if len(idx) > 0:
                print("Warning: duplicate entry found, deleting the old entry:")
                print(df.loc[idx])
                df = df.drop(idx)
            print(f"Appending new entry:")
            new_entry = {**key_values, 'recall_1': recall_1_full, 'recall_10': recall_10_full, 'node_counter': node_counter_full, 'node_counter_per_query': node_counter_per_query_full}
            df = df.append(new_entry, ignore_index=True)
            
            # Save the sub-graph results to the dataframe
            key_values = {
                'dataset': dbname,
                'max_degree': args.MD,
                'mode': 'sub',
                'sub_graph_num': sub_graph_num,
                'ef': ef
            }
            
            idx = df.index[(df['dataset'] == dbname) & \
                            (df['max_degree'] == args.MD) & \
                            (df['mode'] == 'sub') & \
                            (df['sub_graph_num'] == sub_graph_num) & \
                            (df['ef'] == ef)]
            if len(idx) > 0:
                print("Warning: duplicate entry found, deleting the old entry:")
                print(df.loc[idx])
                df = df.drop(idx)
            print(f"Appending new entry:")
            new_entry = {**key_values, 'recall_1': recall_1_sub, 'recall_10': recall_10_sub, 'node_counter': node_counter_sub, 'node_counter_per_query': node_counter_per_query_sub}
            df = df.append(new_entry, ignore_index=True)

        df.to_pickle(perf_df_path)
        
        
        
        
    # print("Sub-graph search recall@1: ", recall_1_sub)
    # print("Full graph search recall@1: ", recall_1_full)
    # print("Sub-graph search recall@10: ", recall_10_sub)
    # print("Full graph search recall@10: ", recall_10_full)
    
    # print("==============================================")
    
    # print("Sub-graph search node counter (per query): ", node_counter_sub)
    # print("Full graph search node counter (per query): ", node_counter_full)    
    