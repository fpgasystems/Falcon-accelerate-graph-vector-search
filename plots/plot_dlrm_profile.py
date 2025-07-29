"""
Assuming CPU retrieval + GPU inference, plot the percentage of time spent on retrieval vs inference.
"""

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import json 

from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns 

plt.rcParams['pdf.fonttype'] = 42
# Seaborn version >= 0.11.0

sns.set_theme(style="whitegrid")
# Set the palette to the "pastel" default palette:
# sns.set_palette("pastel")

# https://seaborn.pydata.org/tutorial/color_palettes.html
# sns.set_palette("Paired")
sns.set_palette("Set2")
# sns.set_palette("husl", 6)

def get_default_colors():

  default_colors = []
  for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
      default_colors.append(color["color"])
      print(color["color"], type(color["color"]))

  return default_colors

default_colors = get_default_colors()

def load_dlrm_latency_json(file_path='./saved_perf_DLRM_GPU/latency_results.json'):
    """Load latency JSON file and return the data as a dictionary."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_retrieval_df(dataset='SIFT1M', graph_type="HNSW", max_degree=64, ef=64, batch_sizes=[1,2,4,8,16], CPU_server="r630", 
                 add_CPU_network_latency=True):

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
    folder_name_CPU_network_latency = './saved_latency/CPU_two_servers'

    d = {}
    d['label'] = []
    d['latency'] = []
    d['category'] = []

    latency_median_CPU = []
    latency_median_FPGA_inter_query = []
    latency_median_FPGA_intra_query = []
    latency_median_FPGA_best = []

    for batch_size in batch_sizes:
            
        """ FPGA Perf """
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
        latency_median_FPGA_best.append(min(latency_median_FPGA_inter_query[-1], latency_median_FPGA_intra_query[-1]))
        
        d['label'].append('batch_size={}'.format(batch_size))
        d['latency'].append(latency_median_FPGA_inter_query[-1])
        d['category'].append('Falcon (Across-query)')
    
        d['label'].append('batch_size={}'.format(batch_size))
        d['latency'].append(latency_median_FPGA_intra_query[-1])
        d['category'].append('Falcon (Intra-query)')

        d['label'].append('batch_size={}'.format(batch_size))
        d['latency'].append(latency_median_FPGA_best[-1])
        d['category'].append('Falcon (Best)')
            
        """ CPU Perf """
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
        latency_median_CPU.append(np.median(df_selected['latency_ms_per_batch'].values[0]) + average_network_latency)
        
        d['label'].append('batch_size={}'.format(batch_size))
        d['latency'].append(latency_median_CPU[-1])
        d['category'].append('CPU (Graph)')

               
    df = pd.DataFrame(data=d)
    
    return df

def plot_latency(
        # Retrieval
        df_retrieval,
        dict_dlrm,
        model_name='RM-S',
        rerank_top_k=16,
        batch_sizes=[1,2,4,8,16],
    ):


    # plt.style.use('ggplot')

    x_labels = [str(i) for i in batch_sizes]
    x = np.arange(len(x_labels))  # the label locations

    bar_lower_retrieval = []
    bar_upper_dlrm = []

    print(f"==== Model {model_name} (rerank top {rerank_top_k}) ====")

    for i, batch_size in enumerate(batch_sizes):

        # Select the appropriate 'data' values for the y-axis based on the batch size
        y_retrieval = df_retrieval[df_retrieval['label'] == 'batch_size={}'.format(batch_size)]['latency'].values
        assert len(y_retrieval) == 1
        y_retrieval = y_retrieval[0] 
        y_dlrm = dict_dlrm[model_name][str(batch_size * rerank_top_k)] * 1000 # sec to ms
        print(f"retrieval : inference (ms) = {y_retrieval:.2f} : {y_dlrm:.2f}")
        perc_retrieval = y_retrieval / (y_retrieval + y_dlrm) * 100
        perc_dlrm = y_dlrm / (y_retrieval + y_dlrm) * 100
        bar_lower_retrieval.append(perc_retrieval)
        bar_upper_dlrm.append(perc_dlrm)

    print(f"Retrieval (%): {np.min(bar_lower_retrieval):.2f} - {np.max(bar_lower_retrieval):.2f}")

    width = 0.35
    fig, ax = plt.subplots(1, 1, figsize=(3,1))
    rects1_lower  = ax.bar(x, bar_lower_retrieval, width)#, label='bar 1 lower')
    rects1_upper  = ax.bar(x, bar_upper_dlrm, width, bottom=bar_lower_retrieval)#, label='bar 1 higher')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Latency (%)', fontsize=11)
    ax.set_xlabel('Batch Size', fontsize=11)
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    
    ax.set_ylim(0, 100)

    ax.legend([rects1_lower, rects1_upper], ["Retrieval", f"{model_name} (rank top {rerank_top_k})"], loc=(-0.3, 1.06), ncol=4, \
        facecolor='white', framealpha=1, frameon=False, fontsize=10)


    def number_single_bar(rects, bottom):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height + bottom[i]),
                        xytext=(0, -20),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    def number_stacked_bar(rects, bottom):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate('{}'.format(height + bottom[i]),
                        xy=(rect.get_x() + rect.get_width() / 2, height + bottom[i]),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    zero_bottom = np.zeros(5)

    # number_single_bar(rects1_lower, zero_bottom)
    # number_single_bar(rects1_upper, bar_lower_retrieval)

    # number_stacked_bar(rects1_upper, bar_lower_retrieval)

    # plt.rcParams.update({'figure.autolayout': True})


    for out_type in ['png', 'pdf']:
        plt.savefig(f'./images/dlrm/dlrm_perc_cpu_{model_name}_rerank_{rerank_top_k}.{out_type}', transparent=False, dpi=200, bbox_inches="tight")

    # plt.show()

def get_e2e_speedup(
        # Retrieval
        df_retrieval_cpu,
        df_retrieval_fpga, 
        dict_dlrm,
        model_name='RM-S',
        rerank_top_k=16,
        batch_sizes=[1,2,4,8,16],
    ):


    e2e_speedup = []

    print(f"==== Model {model_name} (rerank top {rerank_top_k}) ====")

    for i, batch_size in enumerate(batch_sizes):

        # Select the appropriate 'data' values for the y-axis based on the batch size
        y_retrieval_cpu = df_retrieval_cpu[df_retrieval_cpu['label'] == 'batch_size={}'.format(batch_size)]['latency'].values
        assert len(y_retrieval_cpu) == 1
        y_retrieval_cpu = y_retrieval_cpu[0] 

        y_retrieval_fpga = df_retrieval_fpga[df_retrieval_fpga['label'] == 'batch_size={}'.format(batch_size)]['latency'].values
        assert len(y_retrieval_fpga) == 1
        y_retrieval_fpga = y_retrieval_fpga[0]

        y_dlrm = dict_dlrm[model_name][str(batch_size * rerank_top_k)] * 1000 # sec to ms

        e2e_baseline = y_dlrm + y_retrieval_cpu
        e2e_fpga = y_dlrm + y_retrieval_fpga
        speedup = e2e_baseline / e2e_fpga
        e2e_speedup.append(speedup)
        
    print(f"E2E Speedup: {np.min(e2e_speedup):.2f} - {np.max(e2e_speedup):.2f} x")

if __name__ == "__main__":

    # CPU_server = "gold" # awesome cloud Intel(R) Xeon(R) Gold 6242 CPU @ 2.80GHz
    CPU_server = "m5.metal"
    # CPU_server = "m5.8xlarge"
    # CPU_server = "r630"
    # CPU_server = "sgs-gpu"

    # datasets = ["SIFT10M", "Deep10M", "SPACEV10M"]
    # graph_types = ["HNSW", "NSG"]
    dataset = "SIFT10M"
    graph_type = "HNSW"
    batch_sizes = [1, 2, 4, 8]

    for model_name in ['RM-S', 'RM-L']:

        for rerank_top_k in [16, 128]:

            df_retrieval_all = get_retrieval_df(
                dataset=dataset, graph_type=graph_type, max_degree=64, ef=64, batch_sizes=batch_sizes, CPU_server=CPU_server, add_CPU_network_latency=True)
            df_retrieval_cpu = df_retrieval_all[df_retrieval_all['category'] == 'CPU (Graph)']
            df_retrieval_fpga = df_retrieval_all[df_retrieval_all['category'] == 'Falcon (Best)']
            dict_dlrm = load_dlrm_latency_json(file_path='./saved_perf_DLRM_GPU/latency_results.json')
            plot_latency(
                # Retrieval
                df_retrieval_cpu,
                dict_dlrm,
                model_name=model_name, 
                rerank_top_k=rerank_top_k,
                batch_sizes=batch_sizes
            )
            get_e2e_speedup(
                df_retrieval_cpu,
                df_retrieval_fpga,
                dict_dlrm,
                model_name=model_name, 
                rerank_top_k=rerank_top_k,
                batch_sizes=batch_sizes
            )
