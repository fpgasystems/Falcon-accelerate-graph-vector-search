import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns 

# Seaborn version >= 0.11.0

sns.set_theme(style="whitegrid")
# Set the palette to the "pastel" default palette:
# sns.set_palette("pastel")

# https://seaborn.pydata.org/tutorial/color_palettes.html
# sns.set_palette("Paired")
sns.set_palette("Set2")
# sns.set_palette("husl", 6)

global_intra_query_median_latency_speedup_over_GPU = []
global_intra_query_median_latency_speedup_over_CPU = []
global_intra_query_median_latency_speedup_over_CPU_Faiss = []
global_intra_query_median_latency_speedup_over_GPU_Faiss = []
global_inter_query_median_latency_speedup_over_GPU = []
global_inter_query_median_latency_speedup_over_CPU = []
global_inter_query_median_latency_speedup_over_CPU_Faiss = []
global_inter_query_median_latency_speedup_over_GPU_Faiss = []

global_intra_query_P95_latency_speedup_over_GPU = []
global_intra_query_P95_latency_speedup_over_CPU = []
global_intra_query_P95_latency_speedup_over_CPU_Faiss = []
global_intra_query_P95_latency_speedup_over_GPU_Faiss = []
global_inter_query_P95_latency_speedup_over_GPU = []
global_inter_query_P95_latency_speedup_over_CPU = []
global_inter_query_P95_latency_speedup_over_CPU_Faiss = []
global_inter_query_P95_latency_speedup_over_GPU_Faiss = []

def get_default_colors():

  default_colors = []
  for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
      default_colors.append(color["color"])
      print(color["color"], type(color["color"]))

  return default_colors

default_colors = get_default_colors()

def get_perf_df(dataset='SIFT1M', graph_type="HNSW", max_degree=64, ef=64, batch_sizes=[1,2,4,8,16], CPU_server="r630", GPU_model="V100",
                 show_FPGA=True, show_CPU=True, show_GPU=True, show_CPU_Faiss=True, show_GPU_Faiss=True, add_CPU_network_latency=True):

    assert CPU_server in ["gold", "r630", "m5.metal", "m5.8xlarge", "sgs-gpu"]
    
    ### Note: For violin graph, a single violin's data must be in the same column
    ###   e.g., given 3 violin plots, each with 100 points, the shape of the array
    ###   should be (100, 3), where the first column is for the first violin and so forth
    # fake up some data


    # Wenqi: flatten the table to a table. It's a dictionary with the key as schema.
    #   The value of each key is an array.
    # label category data
    # xxx   xxx      xxx
    # yyy   yyy      yyy

    folder_name_FPGA_inter_query = 'saved_latency/FPGA_inter_query_v1_3_4_chan'
    folder_name_FPGA_intra_query = 'saved_latency/FPGA_intra_query_v1_5_4_chan'
    folder_name_CPU = './saved_perf_CPU'
    folder_name_GPU = './saved_perf_GPU'
    folder_name_CPU_network_latency = './saved_latency/CPU_two_servers'

    d = {}
    d['label'] = []
    d['data'] = []
    d['category'] = []
    colors = {}
    markers = {}

    recall_1_CPU = None
    recall_10_CPU = None

    latency_median_CPU = []
    latency_median_GPU = []
    latency_median_FPGA_inter_query = []
    latency_median_FPGA_intra_query = []
    latency_median_CPU_faiss = []
    latency_median_GPU_faiss = []
    
    latency_P95_CPU = []
    latency_P95_GPU = []
    latency_P95_FPGA_inter_query = []
    latency_P95_FPGA_intra_query = []
    latency_P95_CPU_faiss = []
    latency_P95_GPU_faiss = []

    for batch_size in batch_sizes:
        if show_FPGA:
            # load latency distribution (in double)
            # file name
            #   std::string out_fname = "latency_ms_per_batch_" + dataset + "_" + graph_type + 
            # 	  "_MD" + std::to_string(max_degree) + "_ef" + std::to_string(ef) + + "_batch_size" + std::to_string(batch_size) + ".double";
            f_name_FPGA_inter_query = os.path.join(folder_name_FPGA_inter_query, 
                "latency_ms_per_batch_" + dataset + "_" + graph_type + "_MD" + str(max_degree) + "_ef" + str(ef) + "_batch_size" + str(batch_size) + ".double")
            f_name_FPGA_intra_query = os.path.join(folder_name_FPGA_intra_query,
                "latency_ms_per_batch_" + dataset + "_" + graph_type + "_MD" + str(max_degree) + "_ef" + str(ef) + "_batch_size" + str(batch_size) + ".double")
            # load as np array
            latency_FPGA_inter_query = np.fromfile(f_name_FPGA_inter_query, dtype=np.float64)
            latency_FPGA_intra_query = np.fromfile(f_name_FPGA_intra_query, dtype=np.float64)

            latency_median_FPGA_inter_query.append(np.median(latency_FPGA_inter_query))
            latency_median_FPGA_intra_query.append(np.median(latency_FPGA_intra_query))
            latency_P95_FPGA_inter_query.append(np.percentile(latency_FPGA_inter_query, 95))
            latency_P95_FPGA_intra_query.append(np.percentile(latency_FPGA_intra_query, 95))
            
            for latency in latency_FPGA_inter_query:
                d['label'].append('batch_size={}'.format(batch_size))
                d['data'].append(latency)
                d['category'].append('Falcon (Across-query)')
            colors['Falcon (Across-query)'] = default_colors[0]
            markers['Falcon (Across-query)'] = 'o'
            
            for latency in latency_FPGA_intra_query:
                d['label'].append('batch_size={}'.format(batch_size))
                d['data'].append(latency)
                d['category'].append('Falcon (Intra-query)')
            colors['Falcon (Intra-query)'] = default_colors[1]
            markers['Falcon (Intra-query)'] = 'o'
            
        if show_CPU or show_GPU or show_CPU_Faiss or show_GPU_Faiss:
            if add_CPU_network_latency:
                if dataset.startswith("SIFT"):
                    f_name_CPU_network = os.path.join(folder_name_CPU_network_latency, "latency_ms_per_batch_SIFT_batch_size" + str(batch_size) + ".double")
                elif dataset.startswith("Deep"):
                    f_name_CPU_network = os.path.join(folder_name_CPU_network_latency, "latency_ms_per_batch_Deep_batch_size" + str(batch_size) + ".double")
                elif dataset.startswith("SPACEV"):
                    f_name_CPU_network = os.path.join(folder_name_CPU_network_latency, "latency_ms_per_batch_SPACEV_batch_size" + str(batch_size) + ".double")
                latency_CPU_network = np.fromfile(f_name_CPU_network, dtype=np.float64)
                average_network_latency = np.mean(latency_CPU_network)
            else:
                average_network_latency = 0

            if show_CPU:
                if graph_type == "HNSW":
                    if CPU_server == "r630":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_hnsw_cpu_r630.pickle")
                    elif CPU_server == "m5.metal":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_hnsw_cpu_m5.metal_48cores.pickle")
                    elif CPU_server == "m5.8xlarge":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_hnsw_cpu_m5.8xlarge.pickle")
                    elif CPU_server == "sgs-gpu":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_hnsw_cpu_sgs-gpu.pickle")
                    elif CPU_server == "gold":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_hnsw_cpu_gold.pickle")
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
                    elif CPU_server == "m5.metal":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_nsg_cpu_m5.metal_48cores.pickle")
                    elif CPU_server == "m5.8xlarge":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_nsg_cpu_m5.8xlarge.pickle")
                    elif CPU_server == "sgs-gpu":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_nsg_cpu_sgs-gpu.pickle")
                    elif CPU_server == "gold":
                        f_name_CPU = os.path.join(folder_name_CPU, "perf_df_nsg_cpu_gold.pickle")
                    df_cpu = pd.read_pickle(f_name_CPU)

                    # key_columns = ['dataset', 'max_degree', 'search_L', 'omp_enable', 'max_cores', 'batch_size']
                    # result_columns = ['recall_1', 'recall_10', 'latency_ms_per_batch', 'qps']

                    # select rows, and assert only one row is selected
                    df_selected = df_cpu[(df_cpu['dataset'] == dataset) & (df_cpu['max_degree'] == max_degree) & (df_cpu['search_L'] == ef) & (df_cpu['batch_size'] == batch_size)]
                    assert len(df_selected) == 1
                    recall_1_CPU = df_selected['recall_1'].values[0]
                    recall_10_CPU = df_selected['recall_10'].values[0]

                # add latency to d
                for latency in df_selected['latency_ms_per_batch'].values[0]:
                    # print(latency)
                    d['label'].append('batch_size={}'.format(batch_size))
                    d['data'].append(latency + average_network_latency)
                    d['category'].append('CPU (Graph)')

                colors['CPU (Graph)'] = default_colors[2]
                markers['CPU (Graph)'] = 'X'
                latency_median_CPU.append(np.median(df_selected['latency_ms_per_batch'].values[0]) + average_network_latency)
                latency_P95_CPU.append(np.percentile(df_selected['latency_ms_per_batch'].values[0], 95) + average_network_latency)
        
            if show_CPU_Faiss:

                # key_columns = ['dataset', 'max_cores', 'batch_size', 'nprobe']
                # result_columns = ['recall_1', 'recall_10', 'latency_ms_per_batch', 'qps']

                assert show_CPU # need to use the CPU recall as a reference
                if CPU_server == "r630":
                    df_cpu_faiss = pd.read_pickle(os.path.join(folder_name_CPU, "perf_df_faiss_cpu_r630.pickle"))
                elif CPU_server == "m5.metal":
                    df_cpu_faiss = pd.read_pickle(os.path.join(folder_name_CPU, "perf_df_faiss_cpu_m5.metal_48cores.pickle"))
                elif CPU_server == "m5.8xlarge":
                    df_cpu_faiss = pd.read_pickle(os.path.join(folder_name_CPU, "perf_df_faiss_cpu_m5.8xlarge.pickle"))
                elif CPU_server == "sgs-gpu":
                    df_cpu_faiss = pd.read_pickle(os.path.join(folder_name_CPU, "perf_df_faiss_cpu_sgs-gpu.pickle"))
                elif CPU_server == "gold":
                    df_cpu_faiss = pd.read_pickle(os.path.join(folder_name_CPU, "perf_df_faiss_cpu_gold.pickle"))
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
                
                colors['CPU (IVF)'] = default_colors[3]
                markers['CPU (IVF)'] = 'X'
                latency_median_CPU_faiss.append(np.median(df_selected['latency_ms_per_batch'].values[0]) + average_network_latency)
                latency_P95_CPU_faiss.append(np.percentile(df_selected['latency_ms_per_batch'].values[0], 95) + average_network_latency)
                # add latency to d
                for latency in df_selected['latency_ms_per_batch'].values[0]:
                    d['label'].append('batch_size={}'.format(batch_size))
                    d['data'].append(latency + average_network_latency)
                    d['category'].append('CPU (IVF)')

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

                    colors['GPU (Graph)'] = default_colors[4]
                    markers['GPU (Graph)'] = "^"
                    latency_median_GPU.append(np.median(df_selected['latency_ms_per_batch'].values[0]) + average_network_latency)
                    latency_P95_GPU.append(np.percentile(df_selected['latency_ms_per_batch'].values[0], 95) + average_network_latency)	
                    # add latency to d
                    for latency in df_selected['latency_ms_per_batch'].values[0]:
                        # print(latency)
                        d['label'].append('batch_size={}'.format(batch_size))
                        d['data'].append(latency + average_network_latency)
                        d['category'].append('GPU (Graph)')
                # else: # add empty data to the df
                #     for batch_size in batch_sizes:
                #         # print(latency)
                #         d['label'].append('batch_size={}'.format(batch_size))
                #         d['data'].append([])
                #         d['category'].append('GPU (Graph)')


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

                colors['GPU (IVF)'] = default_colors[5]
                markers['GPU (IVF)'] = "^"

                latency_median_GPU_faiss.append(np.median(df_selected['latency_ms_per_batch'].values[0]) + average_network_latency)
                latency_P95_GPU_faiss.append(np.percentile(df_selected['latency_ms_per_batch'].values[0], 95) + average_network_latency)
                # add latency to d
                for latency in df_selected['latency_ms_per_batch'].values[0]:
                    d['label'].append('batch_size={}'.format(batch_size))
                    d['data'].append(latency + average_network_latency)
                    d['category'].append('GPU (IVF)')
                

    df = pd.DataFrame(data=d)
    # print(df.index)
    # print(df.columns)

    print(f"==== {dataset}, {graph_type} ====")
    # print(f"Median Latency (ms):")
    # print(f"CPU: {latency_median_CPU}")
    # if graph_type == "HNSW":
    #     print(f"GPU: {latency_median_GPU}")
    # print(f"FPGA Across-query Parallel: {latency_median_FPGA_inter_query}")
    # print(f"FPGA Intra-query Parallel: {latency_median_FPGA_intra_query}")

    # print(f"P95 Latency (ms):")
    # print(f"CPU: {latency_P95_CPU}")
    # if graph_type == "HNSW":
    #     print(f"GPU: {latency_P95_GPU}")
    # print(f"FPGA Across-query Parallel: {latency_P95_FPGA_inter_query}")
    # print(f"FPGA Intra-query Parallel: {latency_P95_FPGA_intra_query}")

    inter_query_median_latency_speedup_over_CPU = np.array(latency_median_CPU) / np.array(latency_median_FPGA_inter_query)
    intra_query_median_latency_speedup_over_CPU = np.array(latency_median_CPU) / np.array(latency_median_FPGA_intra_query)
    inter_query_P95_latency_speedup_over_CPU = np.array(latency_P95_CPU) / np.array(latency_P95_FPGA_inter_query)
    intra_query_P95_latency_speedup_over_CPU = np.array(latency_P95_CPU) / np.array(latency_P95_FPGA_intra_query)

    inter_query_median_latency_speedup_over_CPU_faiss = np.array(latency_median_CPU_faiss) / np.array(latency_median_FPGA_inter_query)
    intra_query_median_latency_speedup_over_CPU_faiss = np.array(latency_median_CPU_faiss) / np.array(latency_median_FPGA_intra_query)
    inter_query_P95_latency_speedup_over_CPU_faiss = np.array(latency_P95_CPU_faiss) / np.array(latency_P95_FPGA_inter_query)
    intra_query_P95_latency_speedup_over_CPU_faiss = np.array(latency_P95_CPU_faiss) / np.array(latency_P95_FPGA_intra_query)

    inter_query_median_latency_speedup_over_GPU_faiss = np.array(latency_median_GPU_faiss) / np.array(latency_median_FPGA_inter_query)
    intra_query_median_latency_speedup_over_GPU_faiss = np.array(latency_median_GPU_faiss) / np.array(latency_median_FPGA_intra_query)
    inter_query_P95_latency_speedup_over_GPU_faiss = np.array(latency_P95_GPU_faiss) / np.array(latency_P95_FPGA_inter_query)
    intra_query_P95_latency_speedup_over_GPU_faiss = np.array(latency_P95_GPU_faiss) / np.array(latency_P95_FPGA_intra_query)

    global_intra_query_median_latency_speedup_over_CPU.extend(intra_query_median_latency_speedup_over_CPU)
    global_inter_query_median_latency_speedup_over_CPU.extend(inter_query_median_latency_speedup_over_CPU)
    global_intra_query_P95_latency_speedup_over_CPU.extend(intra_query_P95_latency_speedup_over_CPU)
    global_inter_query_P95_latency_speedup_over_CPU.extend(inter_query_P95_latency_speedup_over_CPU)

    global_intra_query_median_latency_speedup_over_CPU_Faiss.extend(intra_query_median_latency_speedup_over_CPU_faiss)
    global_inter_query_median_latency_speedup_over_CPU_Faiss.extend(inter_query_median_latency_speedup_over_CPU_faiss)
    global_intra_query_P95_latency_speedup_over_CPU_Faiss.extend(intra_query_P95_latency_speedup_over_CPU_faiss)
    global_inter_query_P95_latency_speedup_over_CPU_Faiss.extend(inter_query_P95_latency_speedup_over_CPU_faiss)

    global_intra_query_median_latency_speedup_over_GPU_Faiss.extend(intra_query_median_latency_speedup_over_GPU_faiss)
    global_inter_query_median_latency_speedup_over_GPU_Faiss.extend(inter_query_median_latency_speedup_over_GPU_faiss)
    global_intra_query_P95_latency_speedup_over_GPU_Faiss.extend(intra_query_P95_latency_speedup_over_GPU_faiss)
    global_inter_query_P95_latency_speedup_over_GPU_Faiss.extend(inter_query_P95_latency_speedup_over_GPU_faiss)

    if graph_type == "HNSW":
        inter_query_median_latency_speedup_over_GPU = np.array(latency_median_GPU) / np.array(latency_median_FPGA_inter_query)
        intra_query_median_latency_speedup_over_GPU = np.array(latency_median_GPU) / np.array(latency_median_FPGA_intra_query)
        inter_query_P95_latency_speedup_over_GPU = np.array(latency_P95_GPU) / np.array(latency_P95_FPGA_inter_query)
        intra_query_P95_latency_speedup_over_GPU = np.array(latency_P95_GPU) / np.array(latency_P95_FPGA_intra_query)
        global_intra_query_median_latency_speedup_over_GPU.extend(intra_query_median_latency_speedup_over_GPU)
        global_inter_query_median_latency_speedup_over_GPU.extend(inter_query_median_latency_speedup_over_GPU)
        global_intra_query_P95_latency_speedup_over_GPU.extend(intra_query_P95_latency_speedup_over_GPU)
        global_inter_query_P95_latency_speedup_over_GPU.extend(inter_query_P95_latency_speedup_over_GPU)
    print("Across-query speedup over CPU (median): {:.2f}~{:.2f}x".format(np.min(inter_query_median_latency_speedup_over_CPU), np.max(inter_query_median_latency_speedup_over_CPU)))
    print(inter_query_median_latency_speedup_over_CPU)
    print("Intra-query speedup over CPU (median): {:.2f}~{:.2f}x".format(np.min(intra_query_median_latency_speedup_over_CPU), np.max(intra_query_median_latency_speedup_over_CPU)))
    print(intra_query_median_latency_speedup_over_CPU)
    print("Across-query speedup over CPU (P95): {:.2f}~{:.2f}x".format(np.min(inter_query_P95_latency_speedup_over_CPU), np.max(inter_query_P95_latency_speedup_over_CPU)))
    # print(inter_query_P95_latency_speedup_over_CPU)
    print("Intra-query speedup over CPU (P95): {:.2f}~{:.2f}x".format(np.min(intra_query_P95_latency_speedup_over_CPU), np.max(intra_query_P95_latency_speedup_over_CPU)))
    # print(intra_query_P95_latency_speedup_over_CPU)

    if graph_type == "HNSW":
        print("Across-query speedup over GPU (median): {:.2f}~{:.2f}x".format(np.min(inter_query_median_latency_speedup_over_GPU), np.max(inter_query_median_latency_speedup_over_GPU)))
        print(inter_query_median_latency_speedup_over_GPU)
        print("Intra-query speedup over GPU (median): {:.2f}~{:.2f}x".format(np.min(intra_query_median_latency_speedup_over_GPU), np.max(intra_query_median_latency_speedup_over_GPU)))
        print(intra_query_median_latency_speedup_over_GPU)
        print("Across-query speedup over GPU (P95): {:.2f}~{:.2f}x".format(np.min(inter_query_P95_latency_speedup_over_GPU), np.max(inter_query_P95_latency_speedup_over_GPU)))
        print("Intra-query speedup over GPU (P95): {:.2f}~{:.2f}x".format(np.min(intra_query_P95_latency_speedup_over_GPU), np.max(intra_query_P95_latency_speedup_over_GPU)))
    
    return df, recall_10_CPU, colors, markers

def plot_latency(datasets=['SIFT1M'], graph_types=["HNSW"], max_degree=64, ef=64, batch_sizes=[1,2,4,8,16], CPU_server="r630",
                 show_FPGA=True, show_CPU=True, show_GPU=True, show_CPU_Faiss=True, show_GPU_Faiss=True, add_CPU_network_latency=True):

    num_datasets = len(datasets)
    num_graph_types = len(graph_types)
    # num_datasets subplots, horizontally, share the y axis
    fig, axs = plt.subplots(num_graph_types, num_datasets, figsize=(8 * num_datasets, 3 * num_graph_types), sharey=True, sharex=True)

    # reshape axs into 2D array, in (num_graph_types, num_datasets)
    if num_datasets == 1 and num_graph_types == 1:
        axs = np.array([[axs]])
    elif num_datasets == 1:
        axs = axs[:, np.newaxis]
    elif num_graph_types == 1:
        axs = axs[np.newaxis, :]
    # print axs shape
    # print(axs.shape)

    # vertical paddings
    plt.subplots_adjust(hspace=0.7, wspace=0.15)


    df_list = []
    recall_10_CPU_list = [] 
    for gid, graph_type in enumerate(graph_types):
        for pid, dataset in enumerate(datasets):
            ax = axs[gid][pid]
            df, recall_10_CPU, colors, markers = get_perf_df(dataset=dataset, graph_type=graph_type, max_degree=max_degree, ef=ef, batch_sizes=batch_sizes, CPU_server=CPU_server,
                                show_FPGA=show_FPGA, show_CPU=show_CPU, show_GPU=show_GPU, show_CPU_Faiss=show_CPU_Faiss, show_GPU_Faiss=show_GPU_Faiss, add_CPU_network_latency=add_CPU_network_latency)
            df_list.append(df)
            recall_10_CPU_list.append(recall_10_CPU)

            # API: https://seaborn.pydata.org/generated/seaborn.violinplot.html
            # inner{“box”, “quartile”, “point”, “stick”, None}, optional
            # ax = sns.violinplot(data=df, scale='area', inner='box', x="label", y="data", hue="category")
            # use box plot
            # ax = sns.boxplot(data=df, x="label", y="data", hue="category", showfliers=False, ax=ax, palette=colors)
            # use point plot
            # errorbar = ('ci', 95)
            errorbar = 'sd'
            # ax = sns.pointplot(data=df, x="label", y="data", errorbar=errorbar, capsize=.05, hue="category", ax=ax, palette=colors)

            # use point plot, but shift x a little bit for each category
            num_labels = len(colors.keys())

            # Set initial parameters based on the graph type
            width = 0.1
            x_start_inter_q = - num_labels / 2

            # Iterate over each category in the colors dictionary
            for lid, x_labels in enumerate(colors.keys()):
                # Filter the dataframe for the current category
                df_selected = df[df['category'] == x_labels]

                x_data = []
                y_data_mean = []

                # yerr:
                # shape(N,): Symmetric +/-values for each data point.
                # shape(2, N): Separate - and + values for each bar. First row contains the lower errors, the second row contains the upper errors.
                y_data_std = []
                y_data_95 = []

                # Iterate over batch sizes to collect x and y data
                for i, batch_size in enumerate(batch_sizes):
                    # Calculate the x position for the current batch size and category
                    x_data.append(i + (x_start_inter_q + lid) * width)

                    # Select the appropriate 'data' values for the y-axis based on the batch size
                    y_data = df_selected[df_selected['label'] == 'batch_size={}'.format(batch_size)]['data'].values
                    y_mean = np.mean(y_data)
                    y_std = np.std(y_data)  # Standard deviation as error
                    y_95_lower = y_mean - np.percentile(y_data, 5)
                    y_95_upper = np.percentile(y_data, 95) - y_mean
                    y_data_mean.append(y_mean)
                    y_data_std.append(y_std)
                    y_data_95.append([y_95_lower, y_95_upper])
        
                # print(x_data, len(x_data))
                # print(y_data_mean, len(y_data_mean))
                # print(y_data_std, len(y_data_std))
                x_data = np.array(x_data)
                y_data_mean = np.array(y_data_mean)
                y_data_std = np.array(y_data_std)
                y_data_95 = np.array(y_data_95).transpose()

                # Plotting with error bars
                ax.errorbar(x_data, y_data_mean, yerr=y_data_95, fmt=markers[x_labels],  color=colors[x_labels], capsize=6, markersize=8)
                ax.plot(x_data, y_data_mean, marker=markers[x_labels], color=colors[x_labels], label=x_labels, markersize=8, linewidth=2)
                # color is a more shallow version of the original color
                # ax.fill_between(x_data, y_data_mean-y_data_std, y_data_mean+y_data_std, color=colors[x_labels], alpha=0.4)
        

                # Create a point plot for the current category
                # sns.pointplot(x=x_data_all, y=y_data_all, errorbar=errorbar, capsize=.05, ax=ax, color=colors[x_labels])

                # ax = sns.pointplot(data=df_selected, x="label", y="data", errorbar=errorbar, capsize=.05, hue="category", ax=ax, palette=colors)



            # assign each category a color


            label_font = 16
            tick_font = 14
            legend_font = 16
            title_font = 15
            
            x_tick_labels = ["{}".format(i) for i in batch_sizes]
            # print(x_tick_labels)
            ax.set_xticks(range(len(batch_sizes)))
            ax.set_xticklabels(x_tick_labels)
            # ax.set_yticklabels(y_tick_labels)
            # plt.yticks(rotation=0)
            # # ax.ticklabel_format(axis='both', style='sci')
            # # ax.set_yscale("log")
            # ax.legend(loc='best', ncol=2, fontsize=legend_font)
            # if pid == 0:
            if dataset.startswith("SIFT"):# 
                if graph_type == "HNSW":
                    ax.legend(loc=(0.3, 1.18), ncol=6, fontsize=legend_font, frameon=False)
                else:
                    ax.legend(loc=(0.5, 1.18), ncol=5, fontsize=legend_font, frameon=False)
            # else: # do not show legend
            #     ax.get_legend().remove()

            ax.tick_params(length=0, top=False, bottom=True, left=True, right=False, 
                labelleft=True, labelbottom=True, labelsize=tick_font)
            if gid == num_graph_types - 1:
                ax.set_xlabel('Batch sizes', fontsize=label_font, labelpad=0)
            else: # no x label
                ax.set_xlabel('')
            ax.set_ylabel('Latency (ms)', fontsize=label_font, labelpad=5)
            title = f'Dataset: {dataset}, Graph: {graph_type}'
            # if recall_1_CPU is not None:
            #     title += f', R@1≥{recall_1_CPU*100:.2f}%'
            if recall_10_CPU is not None:
                title += f', R@10≥{recall_10_CPU*100:.2f}%'

            ax.set_title(title, fontsize=title_font)
            # plt.text(2, len(y_tick_labels) + 2, "Linear Heatmap", fontsize=16)

            # set y log scale
            ax.set_yscale("log")
            # set y lim
            ax.set_ylim([0.1, 10])
            # ax.set_ylim([0.08, 20])
            # show y ticks on 0.1, 1, 10, 100
            # ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
            ax.yaxis.set_ticks([0.1, 1, 10])

            # show only horizontal grids
            # ax.grid(axis='y', linestyle='-')
            # ax.grid(axis='x', linestyle='-')
            ax.yaxis.grid(True, linestyle='-', which='major')
            ax.xaxis.grid(False)
            
            # read color of the frame
            frame_color = None
            for spine in ax.spines.values():
                if frame_color is None:
                    frame_color = spine.get_edgecolor()
                
            # add vertical line between different x labels
            for i in range(1, len(batch_sizes)):
                ax.axvline(x=i - 0.5, color=frame_color, linewidth=1)
            
            # # if dataset is SIFT and graph is HNSW, annotate the batch size of 1 with Inter/intra-query parallel
            # # if pid == 0:
            # if dataset.startswith("SIFT"):
            #     if graph_type == "HNSW":
            #         width=0.13
            #         x_start_inter_q = -2.5
            #         x_start_intra_q = -1.5
            #     elif graph_type == "NSG":
            #         width=0.15
            #         x_start_inter_q = -2
            #         x_start_intra_q = -1
            #     ax.annotate('Across-q', xy=(x_start_inter_q * width, 0.3), xytext=(x_start_inter_q * width, 1),
            #         arrowprops=dict(facecolor='black', shrink=0., width=3, headwidth=7, frac=0.1), fontsize=label_font, rotation=90, verticalalignment='bottom', horizontalalignment='center')
            #     ax.annotate('Intra-q', xy=(x_start_intra_q * width, 0.2), xytext=(x_start_intra_q * width, 1),
            #         arrowprops=dict(facecolor='black', shrink=0., width=3, headwidth=7, frac=0.1), fontsize=label_font, rotation=90, verticalalignment='bottom', horizontalalignment='center')

    datasets_str = "_".join(datasets)
    graph_types_str = "_".join(graph_types)
    for out_type in ['png', 'pdf']:
        plt.savefig(f'./images/latency_CPU_GPU_FPGA/latency_CPU_GPU_FPGA_{graph_types_str}_{datasets_str}.{out_type}', transparent=False, dpi=200, bbox_inches="tight")

    # plt.show()

if __name__ == "__main__":

    # CPU_server = "gold" # awesome cloud Intel(R) Xeon(R) Gold 6242 CPU @ 2.80GHz
    CPU_server = "m5.metal"
    # CPU_server = "m5.8xlarge"
    # CPU_server = "r630"
    # CPU_server = "sgs-gpu"

    datasets = ["SIFT10M", "Deep10M", "SPACEV10M"]
    graph_types = ["HNSW", "NSG"]
    # datasets = ["SIFT10M"]
    # graph_types = ["HNSW"]
    batch_sizes = [1, 2, 4, 8, 16]

    for graph_type in graph_types:
        plot_latency(datasets, [graph_type], CPU_server=CPU_server, batch_sizes=batch_sizes, show_FPGA=True, show_CPU=True, show_GPU=True, add_CPU_network_latency=True)
    # plot_latency(datasets, graph_types, CPU_server=CPU_server, batch_sizes=batch_sizes, show_FPGA=True, show_CPU=True, show_GPU=True, add_CPU_network_latency=True)

    print("\n\n===== Speedup across all experiments =====")
    print("Across-query latency speedup over CPU (Graph): median: {:.2f}~{:.2f}x P95: {:.2f}~{:.2f}x".format(
        np.min(global_inter_query_median_latency_speedup_over_CPU), np.max(global_inter_query_median_latency_speedup_over_CPU), np.min(global_inter_query_P95_latency_speedup_over_CPU), np.max(global_inter_query_P95_latency_speedup_over_CPU)))
    print("Intra-query latency speedup over CPU (Graph): median: {:.2f}~{:.2f}x P95: {:.2f}~{:.2f}x".format(
        np.min(global_intra_query_median_latency_speedup_over_CPU), np.max(global_intra_query_median_latency_speedup_over_CPU), np.min(global_intra_query_P95_latency_speedup_over_CPU), np.max(global_intra_query_P95_latency_speedup_over_CPU)))

    if "HNSW" in graph_types:
        print("Across-query latency speedup over GPU (Graph): median: {:.2f}~{:.2f}x P95: {:.2f}~{:.2f}x".format(
            np.min(global_inter_query_median_latency_speedup_over_GPU), np.max(global_inter_query_median_latency_speedup_over_GPU),
            np.min(global_inter_query_P95_latency_speedup_over_GPU), np.max(global_inter_query_P95_latency_speedup_over_GPU)))
        print("Intra-query latency speedup over GPU (Graph): median: {:.2f}~{:.2f}x P95: {:.2f}~{:.2f}x".format(
            np.min(global_intra_query_median_latency_speedup_over_GPU), np.max(global_intra_query_median_latency_speedup_over_GPU),
            np.min(global_intra_query_P95_latency_speedup_over_GPU), np.max(global_intra_query_P95_latency_speedup_over_GPU)))

    # take the max speedup of two array for each element
    Falcon_median_speedup_over_CPU = np.maximum(global_inter_query_median_latency_speedup_over_CPU, global_intra_query_median_latency_speedup_over_CPU)
    Falcon_P95_speedup_over_CPU = np.maximum(global_inter_query_P95_latency_speedup_over_CPU, global_intra_query_P95_latency_speedup_over_CPU)
    print("Falcon (best of inter/intra-query) latency speedup over CPU (Graph): median: {:.2f}~{:.2f}x P95: {:.2f}~{:.2f}x".format(
        np.min(Falcon_median_speedup_over_CPU), np.max(Falcon_median_speedup_over_CPU), np.min(Falcon_P95_speedup_over_CPU), np.max(Falcon_P95_speedup_over_CPU)))
    
    if "HNSW" in graph_types:
        Falcon_median_speedup_over_GPU = np.maximum(global_inter_query_median_latency_speedup_over_GPU, global_intra_query_median_latency_speedup_over_GPU)
        Falcon_P95_speedup_over_GPU = np.maximum(global_inter_query_P95_latency_speedup_over_GPU, global_intra_query_P95_latency_speedup_over_GPU)
        print("Falcon (best of inter/intra-query) latency speedup over GPU (Graph): median: {:.2f}~{:.2f}x P95: {:.2f}~{:.2f}x".format(
            np.min(Falcon_median_speedup_over_GPU), np.max(Falcon_median_speedup_over_GPU), np.min(Falcon_P95_speedup_over_GPU), np.max(Falcon_P95_speedup_over_GPU)))

    # print max speedup over CPU/GPU Faiss
    Falcon_median_speedup_over_CPU_Faiss = np.maximum(global_inter_query_median_latency_speedup_over_CPU_Faiss, global_intra_query_median_latency_speedup_over_CPU_Faiss)
    Falcon_P95_speedup_over_CPU_Faiss = np.maximum(global_inter_query_P95_latency_speedup_over_CPU_Faiss, global_intra_query_P95_latency_speedup_over_CPU_Faiss)
    print("Falcon (best of inter/intra-query) latency speedup over CPU (Faiss): median: {:.2f}~{:.2f}x P95: {:.2f}~{:.2f}x".format(
        np.min(Falcon_median_speedup_over_CPU_Faiss), np.max(Falcon_median_speedup_over_CPU_Faiss), np.min(Falcon_P95_speedup_over_CPU_Faiss), np.max(Falcon_P95_speedup_over_CPU_Faiss)))
    Falcon_median_speedup_over_GPU_Faiss = np.maximum(global_inter_query_median_latency_speedup_over_GPU_Faiss, global_intra_query_median_latency_speedup_over_GPU_Faiss)
    Falcon_P95_speedup_over_GPU_Faiss = np.maximum(global_inter_query_P95_latency_speedup_over_GPU_Faiss, global_intra_query_P95_latency_speedup_over_GPU_Faiss)
    print("Falcon (best of inter/intra-query) latency speedup over GPU (Faiss): median: {:.2f}~{:.2f}x P95: {:.2f}~{:.2f}x".format(
        np.min(Falcon_median_speedup_over_GPU_Faiss), np.max(Falcon_median_speedup_over_GPU_Faiss), np.min(Falcon_P95_speedup_over_GPU_Faiss), np.max(Falcon_P95_speedup_over_GPU_Faiss)))
    