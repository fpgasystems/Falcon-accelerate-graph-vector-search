


import numpy as np
import os
import pandas as pd

# key columns: ['graph_type', 'dataset', 'batch_size'], value columns: ['power_in_watt']
key_columns_graph = ['graph_type', 'dataset', 'batch_size']
key_columns_faiss = ['dataset', 'batch_size']
value_columns_graph = ['power_in_watt']
columns_graph = key_columns_graph + value_columns_graph
columns_faiss = key_columns_faiss + value_columns_graph

efficiency_over_CPU = {}
efficiency_over_GPU = {}
efficiency_over_CPU_faiss = {}
efficiency_over_GPU_faiss = {}

def print_energy(dataset='SIFT10M', graph_type="HNSW", max_degree=64, ef=64, batch_sizes=[1,16,10000], CPU_server="m5.metal", GPU_model="V100",
                 show_FPGA=True, show_CPU=True, show_GPU=True, show_CPU_Faiss=True, show_GPU_Faiss=True):

    assert CPU_server in ["r630", "m5.metal"]
    assert GPU_model in ["3090", "V100"]

    if CPU_server == "r630":
        """
        CPU (r630):

        File: log_energy_cpu_hnsw_Deep10M_batch_1       Average energy consumption: 72.43 W
        File: log_energy_cpu_hnsw_Deep10M_batch_10000   Average energy consumption: 110.57 W
        File: log_energy_cpu_hnsw_Deep10M_batch_16      Average energy consumption: 106.68 W
        File: log_energy_cpu_hnsw_SIFT10M_batch_1       Average energy consumption: 66.95 W
        File: log_energy_cpu_hnsw_SIFT10M_batch_10000   Average energy consumption: 109.46 W
        File: log_energy_cpu_hnsw_SIFT10M_batch_16      Average energy consumption: 105.88 W
        File: log_energy_cpu_hnsw_SPACEV10M_batch_1     Average energy consumption: 66.83 W
        File: log_energy_cpu_hnsw_SPACEV10M_batch_10000 Average energy consumption: 106.49 W
        File: log_energy_cpu_hnsw_SPACEV10M_batch_16    Average energy consumption: 100.38 W
        File: log_energy_cpu_nsg_Deep10M_batch_1        Average energy consumption: 66.46 W
        File: log_energy_cpu_nsg_Deep10M_batch_10000    Average energy consumption: 110.55 W
        File: log_energy_cpu_nsg_Deep10M_batch_16       Average energy consumption: 106.28 W
        File: log_energy_cpu_nsg_SIFT10M_batch_1        Average energy consumption: 66.59 W
        File: log_energy_cpu_nsg_SIFT10M_batch_10000    Average energy consumption: 111.25 W
        File: log_energy_cpu_nsg_SIFT10M_batch_16       Average energy consumption: 104.23 W
        File: log_energy_cpu_nsg_SPACEV10M_batch_1      Average energy consumption: 66.75 W
        File: log_energy_cpu_nsg_SPACEV10M_batch_10000  Average energy consumption: 109.99 W
        File: log_energy_cpu_nsg_SPACEV10M_batch_16     Average energy consumption: 104.74 W
        """
        df_power_CPU_graph = pd.DataFrame(columns=columns_graph)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'Deep10M', 'batch_size': 1, 'power_in_watt': 72.43}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'Deep10M', 'batch_size': 10000, 'power_in_watt': 110.57}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'Deep10M', 'batch_size': 16, 'power_in_watt': 106.68}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SIFT10M', 'batch_size': 1, 'power_in_watt': 66.95}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SIFT10M', 'batch_size': 10000, 'power_in_watt': 109.46}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SIFT10M', 'batch_size': 16, 'power_in_watt': 105.88}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SPACEV10M', 'batch_size': 1, 'power_in_watt': 66.83}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SPACEV10M', 'batch_size': 10000, 'power_in_watt': 106.49}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SPACEV10M', 'batch_size': 16, 'power_in_watt': 100.38}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'Deep10M', 'batch_size': 1, 'power_in_watt': 66.46}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'Deep10M', 'batch_size': 10000, 'power_in_watt': 110.55}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'Deep10M', 'batch_size': 16, 'power_in_watt': 106.28}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'SIFT10M', 'batch_size': 1, 'power_in_watt': 66.59}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'SIFT10M', 'batch_size': 10000, 'power_in_watt': 111.25}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'SIFT10M', 'batch_size': 16, 'power_in_watt': 104.23}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'SPACEV10M', 'batch_size': 1, 'power_in_watt': 66.75}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'SPACEV10M', 'batch_size': 10000, 'power_in_watt': 109.99}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'SPACEV10M', 'batch_size': 16, 'power_in_watt': 104.74}, ignore_index=True)

        """
        CPU (r630):

        File: log_energy_cpu_faiss_Deep10M_batch_1      Average energy consumption: 75.62 W
        File: log_energy_cpu_faiss_Deep10M_batch_10000  Average energy consumption: 109.45 W
        File: log_energy_cpu_faiss_Deep10M_batch_16     Average energy consumption: 128.98 W
        File: log_energy_cpu_faiss_SIFT10M_batch_1      Average energy consumption: 75.56 W
        File: log_energy_cpu_faiss_SIFT10M_batch_10000  Average energy consumption: 105.22 W
        File: log_energy_cpu_faiss_SIFT10M_batch_16     Average energy consumption: 129.99 W
        File: log_energy_cpu_faiss_SPACEV10M_batch_1    Average energy consumption: 78.73 W
        File: log_energy_cpu_faiss_SPACEV10M_batch_10000        Average energy consumption: 105.02 W
        File: log_energy_cpu_faiss_SPACEV10M_batch_16   Average energy consumption: 130.40 W
        """
        df_power_CPU_faiss = pd.DataFrame(columns=columns_faiss)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'Deep10M', 'batch_size': 1, 'power_in_watt': 75.62}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'Deep10M', 'batch_size': 10000, 'power_in_watt': 109.45}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'Deep10M', 'batch_size': 16, 'power_in_watt': 128.98}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'SIFT10M', 'batch_size': 1, 'power_in_watt': 75.56}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'SIFT10M', 'batch_size': 10000, 'power_in_watt': 105.22}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'SIFT10M', 'batch_size': 16, 'power_in_watt': 129.99}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'SPACEV10M', 'batch_size': 1, 'power_in_watt': 78.73}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'SPACEV10M', 'batch_size': 10000, 'power_in_watt': 105.02}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'SPACEV10M', 'batch_size': 16, 'power_in_watt': 130.40}, ignore_index=True)

    elif CPU_server == "m5.metal":
        """
        CPU (m5.metal 48 cores, 10000 batch size using 16/48 cores): 
        File: log_energy_cpu_hnsw_Deep10M_batch_1       Average energy consumption: 137.97 W
        File: log_energy_cpu_hnsw_Deep10M_batch_10000   Average energy consumption: 322.40 W
        File: log_energy_cpu_hnsw_Deep10M_batch_16      Average energy consumption: 202.43 W
        File: log_energy_cpu_hnsw_SIFT10M_batch_1       Average energy consumption: 136.93 W
        File: log_energy_cpu_hnsw_SIFT10M_batch_10000   Average energy consumption: 314.01 W
        File: log_energy_cpu_hnsw_SIFT10M_batch_16      Average energy consumption: 201.61 W
        File: log_energy_cpu_hnsw_SPACEV10M_batch_1     Average energy consumption: 138.55 W
        File: log_energy_cpu_hnsw_SPACEV10M_batch_10000 Average energy consumption: 344.51 W
        File: log_energy_cpu_hnsw_SPACEV10M_batch_16    Average energy consumption: 209.26 W
        File: log_energy_cpu_nsg_Deep10M_batch_1        Average energy consumption: 136.71 W
        File: log_energy_cpu_nsg_Deep10M_batch_10000    Average energy consumption: 321.89 W
        File: log_energy_cpu_nsg_Deep10M_batch_16       Average energy consumption: 190.78 W
        File: log_energy_cpu_nsg_SIFT10M_batch_1        Average energy consumption: 136.51 W
        File: log_energy_cpu_nsg_SIFT10M_batch_10000    Average energy consumption: 308.46 W
        File: log_energy_cpu_nsg_SIFT10M_batch_16       Average energy consumption: 192.94 W
        File: log_energy_cpu_nsg_SPACEV10M_batch_1      Average energy consumption: 136.49 W
        File: log_energy_cpu_nsg_SPACEV10M_batch_10000  Average energy consumption: 319.15 W
        File: log_energy_cpu_nsg_SPACEV10M_batch_16     Average energy consumption: 192.46 W
        """
        df_power_CPU_graph = pd.DataFrame(columns=columns_graph)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'Deep10M', 'batch_size': 1, 'power_in_watt': 137.97}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'Deep10M', 'batch_size': 10000, 'power_in_watt': 322.40}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'Deep10M', 'batch_size': 16, 'power_in_watt': 202.43}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SIFT10M', 'batch_size': 1, 'power_in_watt': 136.93}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SIFT10M', 'batch_size': 10000, 'power_in_watt': 314.01}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SIFT10M', 'batch_size': 16, 'power_in_watt': 201.61}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SPACEV10M', 'batch_size': 1, 'power_in_watt': 138.55}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SPACEV10M', 'batch_size': 10000, 'power_in_watt': 344.51}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SPACEV10M', 'batch_size': 16, 'power_in_watt': 209.26}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'Deep10M', 'batch_size': 1, 'power_in_watt': 136.71}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'Deep10M', 'batch_size': 10000, 'power_in_watt': 321.89}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'Deep10M', 'batch_size': 16, 'power_in_watt': 190.78}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'SIFT10M', 'batch_size': 1, 'power_in_watt': 136.51}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'SIFT10M', 'batch_size': 10000, 'power_in_watt': 308.46}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'SIFT10M', 'batch_size': 16, 'power_in_watt': 192.94}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'SPACEV10M', 'batch_size': 1, 'power_in_watt': 136.49}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'SPACEV10M', 'batch_size': 10000, 'power_in_watt': 319.15}, ignore_index=True)
        df_power_CPU_graph = df_power_CPU_graph.append({'graph_type': 'NSG', 'dataset': 'SPACEV10M', 'batch_size': 16, 'power_in_watt': 192.46}, ignore_index=True)

        """
        CPU (m5.metal 48 cores, 10000 batch size using 16/48 cores): 
        File: log_energy_cpu_faiss_Deep10M_batch_1      Average energy consumption: 140.49 W
        File: log_energy_cpu_faiss_Deep10M_batch_10000  Average energy consumption: 305.73 W
        File: log_energy_cpu_faiss_Deep10M_batch_16     Average energy consumption: 217.16 W
        File: log_energy_cpu_faiss_SIFT10M_batch_1      Average energy consumption: 139.20 W
        File: log_energy_cpu_faiss_SIFT10M_batch_10000  Average energy consumption: 306.71 W
        File: log_energy_cpu_faiss_SIFT10M_batch_16     Average energy consumption: 209.53 W
        File: log_energy_cpu_faiss_SPACEV10M_batch_1    Average energy consumption: 140.03 W
        File: log_energy_cpu_faiss_SPACEV10M_batch_10000        Average energy consumption: 305.53 W
        File: log_energy_cpu_faiss_SPACEV10M_batch_16   Average energy consumption: 215.35 W
        """
        df_power_CPU_faiss = pd.DataFrame(columns=columns_faiss)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'Deep10M', 'batch_size': 1, 'power_in_watt': 140.49}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'Deep10M', 'batch_size': 10000, 'power_in_watt': 305.73}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'Deep10M', 'batch_size': 16, 'power_in_watt': 217.16}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'SIFT10M', 'batch_size': 1, 'power_in_watt': 139.20}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'SIFT10M', 'batch_size': 10000, 'power_in_watt': 306.71}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'SIFT10M', 'batch_size': 16, 'power_in_watt': 209.53}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'SPACEV10M', 'batch_size': 1, 'power_in_watt': 140.03}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'SPACEV10M', 'batch_size': 10000, 'power_in_watt': 305.53}, ignore_index=True)
        df_power_CPU_faiss = df_power_CPU_faiss.append({'dataset': 'SPACEV10M', 'batch_size': 16, 'power_in_watt': 215.35}, ignore_index=True)
        
    if GPU_model == "3090":
        """
        GPU (3090):
        File: log_GPU_3090/log_energy_gpu_ggnn_Deep10M_batch_1  Average energy consumption: 153.43
        File: log_GPU_3090/log_energy_gpu_ggnn_Deep10M_batch_10000      Average energy consumption: 314.70
        File: log_GPU_3090/log_energy_gpu_ggnn_Deep10M_batch_16 Average energy consumption: 157.01
        File: log_GPU_3090/log_energy_gpu_ggnn_SIFT10M_batch_1  Average energy consumption: 151.29
        File: log_GPU_3090/log_energy_gpu_ggnn_SIFT10M_batch_10000      Average energy consumption: 290.84
        File: log_GPU_3090/log_energy_gpu_ggnn_SIFT10M_batch_16 Average energy consumption: 156.88
        File: log_GPU_3090/log_energy_gpu_ggnn_SPACEV10M_batch_1        Average energy consumption: 155.10
        File: log_GPU_3090/log_energy_gpu_ggnn_SPACEV10M_batch_10000    Average energy consumption: 291.52
        File: log_GPU_3090/log_energy_gpu_ggnn_SPACEV10M_batch_16       Average energy consumption: 155.59
        """
        df_power_GPU_graph = pd.DataFrame(columns=columns_graph)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'Deep10M', 'batch_size': 1, 'power_in_watt': 153.43}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'Deep10M', 'batch_size': 10000, 'power_in_watt': 314.70}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'Deep10M', 'batch_size': 16, 'power_in_watt': 157.01}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SIFT10M', 'batch_size': 1, 'power_in_watt': 151.29}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SIFT10M', 'batch_size': 10000, 'power_in_watt': 290.84}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SIFT10M', 'batch_size': 16, 'power_in_watt': 156.88}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SPACEV10M', 'batch_size': 1, 'power_in_watt': 155.10}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SPACEV10M', 'batch_size': 10000, 'power_in_watt': 291.52}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SPACEV10M', 'batch_size': 16, 'power_in_watt': 155.59}, ignore_index=True)

        """
        GPU (3090):

        File: log_GPU_3090/log_energy_gpu_faiss_Deep10M_batch_1 Average energy consumption: 188.59
        File: log_GPU_3090/log_energy_gpu_faiss_Deep10M_batch_10000     Average energy consumption: 342.67
        File: log_GPU_3090/log_energy_gpu_faiss_Deep10M_batch_16        Average energy consumption: 324.18
        File: log_GPU_3090/log_energy_gpu_faiss_SIFT10M_batch_1 Average energy consumption: 183.49
        File: log_GPU_3090/log_energy_gpu_faiss_SIFT10M_batch_10000     Average energy consumption: 347.30
        File: log_GPU_3090/log_energy_gpu_faiss_SIFT10M_batch_16        Average energy consumption: 270.53
        File: log_GPU_3090/log_energy_gpu_faiss_SPACEV10M_batch_1       Average energy consumption: 181.77
        File: log_GPU_3090/log_energy_gpu_faiss_SPACEV10M_batch_10000   Average energy consumption: 339.18
        File: log_GPU_3090/log_energy_gpu_faiss_SPACEV10M_batch_16      Average energy consumption: 286.72
        """
        df_power_GPU_faiss = pd.DataFrame(columns=columns_faiss)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'Deep10M', 'batch_size': 1, 'power_in_watt': 188.59}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'Deep10M', 'batch_size': 10000, 'power_in_watt': 342.67}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'Deep10M', 'batch_size': 16, 'power_in_watt': 324.18}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'SIFT10M', 'batch_size': 1, 'power_in_watt': 183.49}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'SIFT10M', 'batch_size': 10000, 'power_in_watt': 347.30}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'SIFT10M', 'batch_size': 16, 'power_in_watt': 270.53}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'SPACEV10M', 'batch_size': 1, 'power_in_watt': 181.77}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'SPACEV10M', 'batch_size': 10000, 'power_in_watt': 339.18}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'SPACEV10M', 'batch_size': 16, 'power_in_watt': 286.72}, ignore_index=True)

    elif GPU_model == "V100":
        
        """
        GPU (V100 16GB):
        File: log_GPU_V100/log_energy_gpu_faiss_Deep10M_batch_1 Average energy consumption: 73.16
        File: log_GPU_V100/log_energy_gpu_faiss_Deep10M_batch_10000     Average energy consumption: 208.37
        File: log_GPU_V100/log_energy_gpu_faiss_Deep10M_batch_16        Average energy consumption: 153.88
        File: log_GPU_V100/log_energy_gpu_faiss_SIFT10M_batch_1 Average energy consumption: 67.26
        File: log_GPU_V100/log_energy_gpu_faiss_SIFT10M_batch_10000     Average energy consumption: 168.08
        File: log_GPU_V100/log_energy_gpu_faiss_SIFT10M_batch_16        Average energy consumption: 116.24
        File: log_GPU_V100/log_energy_gpu_faiss_SPACEV10M_batch_1       Average energy consumption: 66.42
        File: log_GPU_V100/log_energy_gpu_faiss_SPACEV10M_batch_10000   Average energy consumption: 176.42
        File: log_GPU_V100/log_energy_gpu_faiss_SPACEV10M_batch_16      Average energy consumption: 125.75
        """
        df_power_GPU_graph = pd.DataFrame(columns=columns_graph)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'Deep10M', 'batch_size': 1, 'power_in_watt': 73.16}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'Deep10M', 'batch_size': 10000, 'power_in_watt': 208.37}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'Deep10M', 'batch_size': 16, 'power_in_watt': 153.88}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SIFT10M', 'batch_size': 1, 'power_in_watt': 67.26}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SIFT10M', 'batch_size': 10000, 'power_in_watt': 168.08}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SIFT10M', 'batch_size': 16, 'power_in_watt': 116.24}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SPACEV10M', 'batch_size': 1, 'power_in_watt': 66.42}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SPACEV10M', 'batch_size': 10000, 'power_in_watt': 176.42}, ignore_index=True)
        df_power_GPU_graph = df_power_GPU_graph.append({'graph_type': 'HNSW', 'dataset': 'SPACEV10M', 'batch_size': 16, 'power_in_watt': 125.75}, ignore_index=True)

        """ 
        GPU (V100 16GB):
        File: log_GPU_V100/log_energy_gpu_ggnn_Deep10M_batch_1  Average energy consumption: 51.00
        File: log_GPU_V100/log_energy_gpu_ggnn_Deep10M_batch_10000      Average energy consumption: 192.26
        File: log_GPU_V100/log_energy_gpu_ggnn_Deep10M_batch_16 Average energy consumption: 52.03
        File: log_GPU_V100/log_energy_gpu_ggnn_SIFT10M_batch_1  Average energy consumption: 52.51
        File: log_GPU_V100/log_energy_gpu_ggnn_SIFT10M_batch_10000      Average energy consumption: 152.51
        File: log_GPU_V100/log_energy_gpu_ggnn_SIFT10M_batch_16 Average energy consumption: 52.98
        File: log_GPU_V100/log_energy_gpu_ggnn_SPACEV10M_batch_1        Average energy consumption: 56.00
        File: log_GPU_V100/log_energy_gpu_ggnn_SPACEV10M_batch_10000    Average energy consumption: 152.75
        File: log_GPU_V100/log_energy_gpu_ggnn_SPACEV10M_batch_16       Average energy consumption: 56.45
        """
        df_power_GPU_faiss = pd.DataFrame(columns=columns_faiss)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'Deep10M', 'batch_size': 1, 'power_in_watt': 51.00}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'Deep10M', 'batch_size': 10000, 'power_in_watt': 192.26}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'Deep10M', 'batch_size': 16, 'power_in_watt': 52.03}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'SIFT10M', 'batch_size': 1, 'power_in_watt': 52.51}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'SIFT10M', 'batch_size': 10000, 'power_in_watt': 152.51}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'SIFT10M', 'batch_size': 16, 'power_in_watt': 52.98}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'SPACEV10M', 'batch_size': 1, 'power_in_watt': 56.00}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'SPACEV10M', 'batch_size': 10000, 'power_in_watt': 152.75}, ignore_index=True)
        df_power_GPU_faiss = df_power_GPU_faiss.append({'dataset': 'SPACEV10M', 'batch_size': 16, 'power_in_watt': 56.45}, ignore_index=True)


    # the largest power consumption across datasets
    power_FPGA_inter = 58.2 # 42.546
    power_FPGA_intra = 62.3 # 37.653       

    folder_name_FPGA_inter_query = 'saved_latency/FPGA_inter_query_v1_3_4_chan'
    folder_name_FPGA_intra_query = 'saved_latency/FPGA_intra_query_v1_5_4_chan'
    folder_name_CPU = './saved_perf_CPU'
    folder_name_GPU = './saved_perf_GPU'

    # energy consumption in milli-Joules
    energy_per_batch_average_CPU = []
    energy_per_batch_average_GPU = []
    energy_per_batch_average_FPGA_inter_query = []
    energy_per_batch_average_FPGA_intra_query = []
    energy_per_batch_average_CPU_faiss = []
    energy_per_batch_average_GPU_faiss = []


    for batch_size in batch_sizes:
        if batch_size not in efficiency_over_CPU.keys():
            efficiency_over_CPU[batch_size] = []
            efficiency_over_GPU[batch_size] = []
            efficiency_over_CPU_faiss[batch_size] = []
            efficiency_over_GPU_faiss[batch_size] = []

        print(f"\n==== {graph_type}, {dataset}, batch size {batch_size} =====")
        if show_FPGA:
            # load latency distribution (in double)
            # file name
            #   std::string out_fname = "latency_ms_per_batch_" + dataset + "_" + graph_type + 
            # 	  "_MD" + std::to_string(max_degree) + "_ef" + std::to_string(ef) + + "_batch_size" + std::to_string(batch_size) + ".double";
            if batch_size > 32:
                batch_size_rewritten = 32
                latency_factor = batch_size / 32
            else: 
                batch_size_rewritten = batch_size
                latency_factor = 1
            f_name_FPGA_inter_query = os.path.join(folder_name_FPGA_inter_query, 
                "latency_ms_per_batch_" + dataset + "_" + graph_type + "_MD" + str(max_degree) + "_ef" + str(ef) + "_batch_size" + str(batch_size_rewritten) + ".double")
            f_name_FPGA_intra_query = os.path.join(folder_name_FPGA_intra_query,
                "latency_ms_per_batch_" + dataset + "_" + graph_type + "_MD" + str(max_degree) + "_ef" + str(ef) + "_batch_size" + str(batch_size_rewritten) + ".double")
            # load as np array
            latency_FPGA_inter_query = np.fromfile(f_name_FPGA_inter_query, dtype=np.float64) * latency_factor
            latency_FPGA_intra_query = np.fromfile(f_name_FPGA_intra_query, dtype=np.float64) * latency_factor

            latency_ms = np.mean(latency_FPGA_inter_query)
            energy_mj = power_FPGA_inter * latency_ms
            energy_per_batch_average_FPGA_inter_query.append(energy_mj)
            print(f"Energy consumption of FPGA inter-query: {energy_mj:.2f} mJ")

            latency_ms = np.mean(latency_FPGA_intra_query)
            energy_mj = power_FPGA_intra * latency_ms
            energy_per_batch_average_FPGA_intra_query.append(energy_mj)
            print(f"Energy consumption of FPGA intra-query: {energy_mj:.2f} mJ")

            best_energy_per_batch_FPGA = min(energy_per_batch_average_FPGA_inter_query[-1], energy_per_batch_average_FPGA_intra_query[-1])
            
        if show_CPU or show_GPU or show_CPU_Faiss or show_GPU_Faiss:

            if show_CPU:
                if graph_type == "HNSW":
                    if CPU_server == "r630":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_hnsw_cpu_r630.pickle")
                    elif CPU_server == "sgs-gpu":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_hnsw_cpu_sgs-gpu.pickle")
                    elif CPU_server == "m5.metal":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_hnsw_cpu_m5.metal_48cores.pickle")
                    df_cpu = pd.read_pickle(f_name_CPU)

                    # key_columns = ['dataset', 'max_degree', 'ef', 'omp_enable', 'max_cores', 'batch_size']
                    # result_columns = ['recall_1', 'recall_10', 'latency_ms_per_batch', 'qps']

                    # select rows, and assert only one row is selected
                    df_selected = df_cpu[(df_cpu['dataset'] == dataset) & (df_cpu['max_degree'] == max_degree) & (df_cpu['ef'] == ef) & (df_cpu['batch_size'] == batch_size)]
                    assert len(df_selected) == 1
                    recall_1_CPU = df_selected['recall_1'].values[0]
                    recall_10_CPU = df_selected['recall_10'].values[0]

                elif graph_type == "NSG":
                    if CPU_server == "r630":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_nsg_cpu_r630.pickle")
                    elif CPU_server == "sgs-gpu":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_nsg_cpu_sgs-gpu.pickle")
                    elif CPU_server == "m5.metal":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_nsg_cpu_m5.metal_48cores.pickle")
                    df_cpu = pd.read_pickle(f_name_CPU)

                    # key_columns = ['dataset', 'max_degree', 'search_L', 'omp_enable', 'max_cores', 'batch_size']
                    # result_columns = ['recall_1', 'recall_10', 'latency_ms_per_batch', 'qps']

                    # select rows, and assert only one row is selected
                    df_selected = df_cpu[(df_cpu['dataset'] == dataset) & (df_cpu['max_degree'] == max_degree) & (df_cpu['search_L'] == ef) & (df_cpu['batch_size'] == batch_size)]
                    assert len(df_selected) == 1
                    recall_1_CPU = df_selected['recall_1'].values[0]
                    recall_10_CPU = df_selected['recall_10'].values[0]

                latency_ms = np.mean(df_selected['latency_ms_per_batch'].values[0])
                power_w = df_power_CPU_graph[(df_power_CPU_graph['graph_type'] == graph_type) & (df_power_CPU_graph['dataset'] == dataset) & (df_power_CPU_graph['batch_size'] == batch_size)]['power_in_watt'].values[0]
                energy_mj = power_w * latency_ms
                energy_per_batch_average_CPU.append(energy_mj)
                energy_ratio_over_FPGA = energy_mj / best_energy_per_batch_FPGA
                efficiency_over_CPU[batch_size].append(energy_ratio_over_FPGA)
                print(f"Energy consumption of CPU: {energy_mj:.2f} mJ\t{energy_ratio_over_FPGA:.2f} x over FPGA")
                
        
            if show_CPU_Faiss:

                # key_columns = ['dataset', 'max_cores', 'batch_size', 'nprobe']
                # result_columns = ['recall_1', 'recall_10', 'latency_ms_per_batch', 'qps']

                assert show_CPU # need to use the CPU recall as a reference
                if CPU_server == "r630":
                    df_cpu_faiss = pd.read_pickle(os.path.join(folder_name_CPU, "perf_df_faiss_cpu_r630.pickle"))
                elif CPU_server == "sgs-gpu":
                    df_cpu_faiss = pd.read_pickle(os.path.join(folder_name_CPU, "perf_df_faiss_cpu_sgs-gpu.pickle"))
                elif CPU_server == "m5.metal":
                    df_cpu_faiss = pd.read_pickle(os.path.join(folder_name_CPU, "perf_df_faiss_cpu_m5.metal_48cores.pickle"))
                df_selected_list = df_cpu_faiss[(df_cpu_faiss['dataset'] == dataset) & (df_cpu_faiss['batch_size'] == batch_size)]
                nprobe_list_sorted = sorted(df_selected_list['nprobe'].values)
                # use the performance of minimum nprobe that can achieve the same recall as CPU graph (recall_10_CPU)
                df_selected = None
                for nprobe in nprobe_list_sorted:
                    recall_10 = df_selected_list[df_selected_list['nprobe'] == nprobe]['recall_10'].values[0]
                    if recall_10 >= recall_10_CPU:
                        df_selected = df_selected_list[df_selected_list['nprobe'] == nprobe]
                        break
                assert len(df_selected) == 1
                
                latency_ms = np.mean(df_selected['latency_ms_per_batch'].values[0])
                power_w = df_power_CPU_faiss[(df_power_CPU_faiss['dataset'] == dataset) & (df_power_CPU_faiss['batch_size'] == batch_size)]['power_in_watt'].values[0]
                energy_mj = power_w * latency_ms
                energy_per_batch_average_CPU_faiss.append(energy_mj)
                energy_ratio_over_FPGA = energy_mj / best_energy_per_batch_FPGA
                efficiency_over_CPU_faiss[batch_size].append(energy_ratio_over_FPGA)
                print(f"Energy consumption of CPU Faiss: {energy_mj:.2f} mJ\t{energy_ratio_over_FPGA:.2f} x over FPGA")

            if show_GPU:
                # GPU only supports HNSW
                if graph_type == "HNSW":
                    if GPU_model == "3090":
                        f_name_GPU = os.path.join(folder_name_GPU, "perf_df_ggnn_gpu_3090.pickle")
                    elif GPU_model == "V100":
                        f_name_GPU = os.path.join(folder_name_GPU, "perf_df_ggnn_gpu_V100.pickle")
                    df_gpu = pd.read_pickle(f_name_GPU)

                    # key_columns = ['dataset', 'KBuild', 'S', 'KQuery', 'MaxIter', 'batch_size', 'tau_query']
                    # result_columns = ['recall_1', 'recall_10', 'latency_ms_per_batch', 'qps']

                    # select rows, and assert only one row is selected
                    degree_gpu = 32 # gpu used full knn graph with prunning, thus average degree ~= max degree
                    max_iter_gpu = 400 # 400 iterations achieves recall near CPU
                    df_selected = df_gpu[(df_gpu['dataset'] == dataset) & (df_gpu['KBuild'] == degree_gpu) & (df_gpu['MaxIter'] == max_iter_gpu) & (df_gpu['batch_size'] == batch_size)]
                    assert len(df_selected) == 1

                    latency_ms = np.mean(df_selected['latency_ms_per_batch'].values[0])
                    power_w = df_power_GPU_graph[(df_power_GPU_graph['graph_type'] == graph_type) & (df_power_GPU_graph['dataset'] == dataset) & (df_power_GPU_graph['batch_size'] == batch_size)]['power_in_watt'].values[0]
                    energy_mj = power_w * latency_ms
                    energy_per_batch_average_GPU.append(energy_mj)
                    energy_ratio_over_FPGA = energy_mj / best_energy_per_batch_FPGA
                    efficiency_over_GPU[batch_size].append(energy_ratio_over_FPGA)
                    print(f"Energy consumption of GPU: {energy_mj:.2f} mJ\t{energy_ratio_over_FPGA:.2f} x over FPGA")

            if show_GPU_Faiss:
               
                # key_columns = ['dataset', 'max_cores', 'batch_size', 'nprobe']
                # result_columns = ['recall_1', 'recall_10', 'latency_ms_per_batch', 'qps'] 
                if GPU_model == "3090":
                    df_gpu_faiss = pd.read_pickle(os.path.join(folder_name_GPU, "perf_df_faiss_gpu_3090.pickle"))
                elif GPU_model == "V100":
                    df_gpu_faiss = pd.read_pickle(os.path.join(folder_name_GPU, "perf_df_faiss_gpu_V100.pickle"))
                df_selected_list = df_gpu_faiss[(df_gpu_faiss['dataset'] == dataset) & (df_gpu_faiss['batch_size'] == batch_size)]
                nprobe_list_sorted = sorted(df_selected_list['nprobe'].values)
                # use the performance of minimum nprobe that can achieve the same recall as CPU graph (recall_10_CPU)
                df_selected = None
                for nprobe in nprobe_list_sorted:
                    recall_10 = df_selected_list[df_selected_list['nprobe'] == nprobe]['recall_10'].values[0]
                    if recall_10 >= recall_10_CPU:
                        df_selected = df_selected_list[df_selected_list['nprobe'] == nprobe]
                        break
                assert len(df_selected) == 1

                latency_ms = np.mean(df_selected['latency_ms_per_batch'].values[0])
                power_w = df_power_GPU_faiss[(df_power_GPU_faiss['dataset'] == dataset) & (df_power_GPU_faiss['batch_size'] == batch_size)]['power_in_watt'].values[0]
                energy_mj = power_w * latency_ms
                energy_per_batch_average_GPU_faiss.append(energy_mj)
                energy_ratio_over_FPGA = energy_mj / best_energy_per_batch_FPGA
                efficiency_over_GPU_faiss[batch_size].append(energy_ratio_over_FPGA)
                print(f"Energy consumption of GPU Faiss: {energy_mj:.2f} mJ\t{energy_ratio_over_FPGA:.2f} x over FPGA")


if __name__ == "__main__":

    CPU_server = "m5.metal"
    # CPU_server = "r630"
    # CPU_server = "sgs-gpu"

    GPU_model = "V100"
    # GPU_model = "3090"

    datasets = ["SIFT10M", "Deep10M", "SPACEV10M"]
    graph_types = ["HNSW", "NSG"]

    batch_sizes = [1, 16, 10000]

    for graph_type in graph_types:
        for dataset in datasets:
            print_energy(dataset=dataset, graph_type=graph_type, batch_sizes=batch_sizes, CPU_server=CPU_server, GPU_model=GPU_model)

    print("\n===== Energy efficiency across datasets =====")
    for batch_size in batch_sizes:
        # show min~max efficiency over each baseline
        print(f"Batch size: {batch_size}")
        print("Efficiency over CPU: {:.2f} ~ {:.2f} x".format(min(efficiency_over_CPU[batch_size]), max(efficiency_over_CPU[batch_size])))
        print("Efficiency over GPU: {:.2f} ~ {:.2f} x".format(min(efficiency_over_GPU[batch_size]), max(efficiency_over_GPU[batch_size])))
        print("Efficiency over CPU Faiss: {:.2f} ~ {:.2f} x".format(min(efficiency_over_CPU_faiss[batch_size]), max(efficiency_over_CPU_faiss[batch_size])))
        print("Efficiency over GPU Faiss: {:.2f} ~ {:.2f} x".format(min(efficiency_over_GPU_faiss[batch_size]), max(efficiency_over_GPU_faiss[batch_size])))
