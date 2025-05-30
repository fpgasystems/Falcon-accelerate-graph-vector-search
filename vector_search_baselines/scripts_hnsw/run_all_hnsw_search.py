'''
Sample command to run the script:
    python run_all_hnsw_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_search_bin_path ../hnswlib/build/main \
        --perf_df_path perf_df.pickle --dataset SIFT1M --max_cores 16 --nruns 3
'''


import os
import argparse
import json
import glob
import pandas as pd
import numpy as np

def read_from_log(log_fname:str):
    """
    Read recall and node_count from the log file
    """
    num_batch = 0
    latency_ms_per_batch = []
    qps = 0.0
    recall_1 = None
    recall_10 = None

    with open(log_fname, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            if "qsize divided into" in line:
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
    parser.add_argument('--perf_df_path', type=str, default='perf_df.pickle')
    parser.add_argument('--hnsw_search_bin_path', type=str, default=None, help='path to algorithm bin')
    parser.add_argument('--hnsw_index_path', type=str, default="../data/CPU_hnsw_indexes", help='path to output constructed nsg index')
    parser.add_argument('--dataset', type=str, default='SIFT1M', help='dataset to use: SIFT1M')
    parser.add_argument('--max_cores', type=int, default=16, help='maximum number of cores used for search')
    parser.add_argument('--nruns', type=int, default=3, help='number of runs per setting (for recording latency and throughput)')

    args = parser.parse_args()

    perf_df_path:str = args.perf_df_path
    hnsw_search_bin_path:str = args.hnsw_search_bin_path
    hnsw_index_path:str = args.hnsw_index_path
    dataset:str = args.dataset
    max_cores:int = args.max_cores
    nruns = args.nruns

    hnsw_index_path = os.path.abspath(hnsw_index_path)

    # random num log
    log_perf_test = 'hnsw_{}.log'.format(np.random.randint(1000000))

    if not os.path.exists(hnsw_search_bin_path):
        raise ValueError(f"Path to algorithm does not exist: {hnsw_search_bin_path}")

    key_columns = ['dataset', 'max_degree', 'ef', 'omp_enable', 'max_cores', 'batch_size']
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

    # Full grid search
    max_degree_list = [64]
    # ef_list = [16]
    ef_list = [64]
    # ef_list = [16, 32, 48, 64, 80, 96]
    omp_list = [1] # 1 = enable; 0 = disable
    batch_size_list = [1, 2, 4, 8, 16, 32, 10000]
    # batch_size_list = [10000]

    for max_degree in max_degree_list:

        hnsw_index_file = os.path.join(hnsw_index_path, f"{dataset}_index_MD{max_degree}.bin")
        assert os.path.exists(hnsw_index_file)

        for ef in ef_list:
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
                        cmd_search_hnsw = f"taskset --cpu-list 0-{max_cores-1} " + \
                                            f"{hnsw_search_bin_path} " + \
                                            f"{dataset} " + \
                                            f"{hnsw_index_file} " + \
                                            f"{ef} " + \
                                            f"{omp} " + \
                                            f"{interq_multithread} " + \
                                            f"{batch_size} " + \
                                            f" > {log_perf_test}"
                        print(f"Searching HNSW by running command:\n{cmd_search_hnsw}")
                        os.system(cmd_search_hnsw)
                        
                        recall_1, recall_10, latency_ms_per_batch_this_run, qps_this_run = read_from_log(log_perf_test)
                        # maintain 1-d list
                        latency_ms_per_batch.extend(latency_ms_per_batch_this_run)
                        qps.append(qps_this_run)
                        
                    key_values = {
                        'dataset': dataset,
                        'max_degree': max_degree,
                        'ef': ef,
                        'omp_enable': omp,
                        'max_cores': max_cores,
                        'batch_size': batch_size
                    }
                    
                    
                    idx = df.index[(df['dataset'] == dataset) & \
                                    (df['max_degree'] == max_degree) & \
                                    (df['ef'] == ef) & \
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

    # os.system("rm " + log_perf_test)