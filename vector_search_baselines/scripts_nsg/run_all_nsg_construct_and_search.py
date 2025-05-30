'''
This script runs c++ nsg to either (a) construct index or (b) measure the recall and search time.

Example Usage (construct):

python run_all_nsg_construct_and_search.py --mode construct \
    --input_knng_path ../data/CPU_knn_graphs --output_nsg_path ../data/CPU_NSG_index \
    --nsg_con_bin_path ../nsg/build/tests/test_nsg_index --dataset SIFT1M

Example Usage (search):

python run_all_nsg_construct_and_search.py --mode search \
    --nsg_search_bin_path ../nsg/build/tests/test_nsg_optimized_search \
    --dataset SIFT1M --perf_df_path perf_df.pickle --max_cores 16 --nruns 3
    
    The latest result can be viewed in temp log: "./nsg.log".
    Currently we support SIFT, Deep, and SBERT1M.
'''
import os
import argparse
import glob
import numpy as np
import pandas as pd

def read_from_log(log_fname:str, mode:str):
    if mode == "construct":
        """
        Read real max degree from the log file
        """
        real_max_degree = 0
        with open(log_fname, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if "Degree Statistics" in line:
                    real_max_degree = int(line.split(" ")[4].split(",")[0])
        return real_max_degree
    else:
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
                    num_batch = int(line.split(" ")[3])
                elif "recall_1:" in line:
                    recall_1 = float(line.split(" ")[1])
                elif "recall_10:" in line:
                    recall_10 = float(line.split(" ")[1])
                elif "time for each batch" in line:
                    # recorded in us
                    for i in range(num_batch):
                        values = lines[idx + 1 + i].split(" ")
                        latency_ms_per_batch.append(float(values[0]) / 1000)
                elif "qps" in line:
                    qps = float(line.split(" ")[1])

        # if more than 2 batches, remove the latency of the first and last batch
        if len(latency_ms_per_batch) > 2:
            latency_ms_per_batch = latency_ms_per_batch[1:-1]	

        return recall_1, recall_10, latency_ms_per_batch, qps


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='search', help='mode to use: either construct or search', choices=['construct', 'search'])
    parser.add_argument('--input_knng_path', type=str, default="../data/CPU_knn_graphs", help='path to input knn graph for index construction')
    parser.add_argument('--output_nsg_path', type=str, default="../data/CPU_NSG_index", help='path to output constructed nsg index')
    parser.add_argument('--perf_df_path', type=str, default='perf_df.pickle', help='path to save the performance dataframe')
    parser.add_argument('--nsg_con_bin_path', type=str, default=None, help='path to nsg construction algorithm')
    parser.add_argument('--nsg_search_bin_path', type=str, default=None, help='path to nsg search algorithm')
    parser.add_argument('--dataset', type=str, default='SIFT1M', help='dataset to use: SIFT1M')
    parser.add_argument('--max_cores', type=int, default=16, help='maximum number of cores used for search')
    parser.add_argument('--nruns', type=int, default=3, help='number of runs per setting (for recording latency and throughput)')

    args = parser.parse_args()

    perf_df_path:str = args.perf_df_path
    nsg_con_bin_path:str = args.nsg_con_bin_path
    nsg_search_bin_path:str = args.nsg_search_bin_path
    input_knng_path = args.input_knng_path
    output_nsg_path = args.output_nsg_path
    dataset:str = args.dataset
    mode = args.mode
    max_cores = args.max_cores
    nruns = args.nruns

    # convert to absolute path
    input_knng_path = os.path.abspath(input_knng_path)
    output_nsg_path = os.path.abspath(output_nsg_path)
    
    # random num log
    log_nsg = 'nsg_{}.log'.format(np.random.randint(1000000))

    if mode == "construct":
          
        if not os.path.exists(nsg_con_bin_path):
            raise ValueError(f"Path to algorithm does not exist: {nsg_con_bin_path}")
    
        construct_K_list_const = [200] # knn graph
        construct_L_list_const = [50] # a magic number that controls graph construction quality
        construct_C_list_const = [500] # another magic number that controls graph construction quality
        construct_R_list = [16, 32, 64] # R means max degree
        
        for construct_K in construct_K_list_const:        
            # Check if KNN graph file exists
            knng_path = os.path.join(input_knng_path, f"{dataset}_{construct_K}NN.graph")
            if not os.path.exists(knng_path):
                print(f"KNN graph file does not exist: {knng_path}, use command:")
                print(f"    python construct_faiss_knn.py --dataset {dataset} --construct_K {construct_K} --output_path {input_knng_path}")
                exit()
            print(f"KNN graph file exists: {knng_path}")
            
            for construct_L in construct_L_list_const:
                for construct_C in construct_C_list_const:
                    for construct_R in construct_R_list:
                        R = construct_R
                        # Construct NSG index
                        if not os.path.exists(output_nsg_path):
                            os.mkdir(output_nsg_path)
                        nsg_file = os.path.join(output_nsg_path, f"{dataset}_index_MD{construct_R}.nsg")
                        if os.path.exists(nsg_file):
                            print(f"NSG index file already exists: {nsg_file}")
                        else:
                            print(f"Generating NSG index file: {nsg_file}")
                            assert os.path.exists(knng_path)
                            # Here: need to manually -1, cause we got this when using 16 as max degree
                            #   Degree Statistics: Max = 17, Min = 1, Avg = 15 (finally push a link to the root)
                            cmd_build_nsg = f"{nsg_con_bin_path} {dataset} {knng_path} {construct_L} {construct_R} {construct_C} {nsg_file} > {log_nsg}"
                            print(f"Constructing NSG by running command:\n{cmd_build_nsg}")
                            os.system(cmd_build_nsg)
                            real_max_degree = read_from_log(log_nsg, mode)
                            
                            if not os.path.exists(nsg_file):
                                print("NSG index file construction failed")
                                exit()
                            print("NSG index file construction succeeded")
                            
                            
                            while real_max_degree > construct_R:
                                cmd_rm_nsg = f"rm {nsg_file}"
                                os.system(cmd_rm_nsg)
                                
                                if construct_R == 0:
                                    print("Max degree is 0, fail to construct the NSG index")
                                    exit()
                                    
                                print(f"Real max degree {real_max_degree} is not equal to the expected max degree {construct_R}, re-construct the NSG index")
                                R -= 1
                                cmd_build_nsg = f"{nsg_con_bin_path} {dataset} {knng_path} {construct_L} {R} {construct_C} {nsg_file} > {log_nsg}"
                                print(f"Constructing NSG by running command:\n{cmd_build_nsg}")
                                os.system(cmd_build_nsg)
                                real_max_degree = read_from_log(log_nsg, mode)
                                if not os.path.exists(nsg_file):
                                    print("NSG index file construction failed")
                                    exit()
                                print("NSG index file construction succeeded")
                            
                            print(f"Succeed to construct the NSG index with max degree {real_max_degree}")
            
            
    elif mode == "search": 

        if not os.path.exists(nsg_search_bin_path):
            raise ValueError(f"Path to algorithm does not exist: {nsg_search_bin_path}")

        key_columns = ['dataset', 'max_degree', 'search_L', 'omp_enable', 'max_cores', 'batch_size']
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
    
        construct_R_list = [64] # R means max degree
        search_L_list = [64]
        # search_L_list = [64, 80, 96]
        # search_L_list = [16, 32, 48, 64, 80, 96]
        omp_list = [1] # 1 = enable; 0 = disable
        # batch_size_list = [1]
        batch_size_list = [1, 2, 4, 8, 16, 32, 10000]
        print("Warning: please set the search parameters in the script, current settings:")
        print(f"construct_R_list: {construct_R_list}")
        print(f"search_L_list: {search_L_list}")	
        print(f"omp_list: {omp_list}")
        print(f"batch_size_list: {batch_size_list}")

        for construct_R in construct_R_list:
            nsg_file = os.path.join(output_nsg_path, f"{dataset}_index_MD{construct_R}.nsg")
            assert os.path.exists(nsg_file)

            # Perform the search
            for search_L in search_L_list:
                for omp in omp_list:
                    for batch_size in batch_size_list:
                        if omp == 0:
                            interq_multithread = 1
                        else:
                            if batch_size > max_cores:
                                interq_multithread = max_cores
                            else:
                                interq_multithread = batch_size
                        
                        latency_ms_per_batch = []
                        qps = []
                        for run in range(nruns):
                            cmd_search_nsg = f"taskset --cpu-list 0-{max_cores-1} " + \
                                            f"{nsg_search_bin_path} " + \
                                            f"{dataset} " + \
                                            f"{nsg_file} " + \
                                            f"{search_L} " + \
                                            f"{omp} " + \
                                            f"{interq_multithread} " + \
                                            f"{batch_size}" + \
                                            f" > {log_nsg}"
                            print(f"Searching NSG by running command:\n{cmd_search_nsg}")
                            os.system(cmd_search_nsg)

                            recall_1, recall_10, latency_ms_per_batch_this_run, qps_this_run = read_from_log(log_nsg, mode)
                            latency_ms_per_batch.extend(latency_ms_per_batch_this_run)
                            qps.append(qps_this_run)

                        # print(f"Recall: {recall}, Time Batch: {latency_ms_per_batch}, QPS: {qps}")
                        # if already in the df, delete the old row first
                        key_values = {
                            'dataset': dataset,
                            'max_degree': construct_R,
                            'search_L': search_L,
                            'omp_enable': omp,
                            'max_cores': max_cores,
                            'batch_size': batch_size
                        }
                        
                        idx = df.index[(df['dataset'] == dataset) & \
                                        (df['max_degree'] == construct_R) & \
                                        (df['search_L'] == search_L) & \
                                        (df['omp_enable'] == omp) & \
                                        (df['max_cores'] == max_cores) & \
                                        (df['batch_size'] == batch_size)]
                        if len(idx) > 0:
                            print("Warning: duplicate entry found, deleting the old entry:")
                            print(df.loc[idx])
                            df = df.drop(idx)
                        print(f"Appending new entry:")
                        new_entry = {**key_values, 'recall_1': recall_1, 'recall_10': recall_10, 'latency_ms_per_batch': latency_ms_per_batch, 'qps': qps}
                        print(new_entry)
                        df = df.append(new_entry, ignore_index=True)
    
        df.to_pickle(args.perf_df_path, protocol=4)

    # os.system("rm " + log_nsg)