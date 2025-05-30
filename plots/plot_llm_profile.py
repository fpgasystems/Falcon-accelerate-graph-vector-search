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


def load_llm_latency_df(
    file_path: str,
):
    """
    Function to load performance data from a CSV file based on the data type ('retrieval' or 'inference').
    Filters out unnecessary columns based on the specified data type.

    Parameters:
    file_path (str): Path to the CSV file.
    data_type (str): The type of data ('retrieval' or 'inference').

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """

    df = df = pd.read_csv(file_path)

    if df is None:
        raise ValueError("Failed to load data. Please check the file path and format.")

    # Define the columns to keep for inference
    inference_columns = [
        "model.name",
        "model.stage",
        "model.dec_steps",
        "model.seq_len_inference_prefill",
        "hardware.name",
        "hardware.num_chips",
        "batch_size",
        "time_ms",
    ]

    # Check if the necessary columns exist in the DataFrame
    missing_columns = [col for col in inference_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns for inference data: {missing_columns}")
    # Keep only the relevant columns for inference
    df = df[inference_columns]


    df['hardware.name'] = df['hardware.name'].replace('V100_32GB', 'V100')
    df['hardware.name'] = df['hardware.name'].replace('A100_80GB_GPU', 'A100')
    df['hardware.name'] = df['hardware.name'].replace('H100_GPU', 'H100')
    
    df['model.name'] = df['model.name'].replace('meta-llama/Llama-3.2-1B', '1B')
    df['model.name'] = df['model.name'].replace('meta-llama/Llama-3.2-3B', '3B')
    df['model.name'] = df['model.name'].replace('meta-llama/llama-2-13b', '13B')

    return df

# def preprocess_llm_latency_df(
#     df: pd.DataFrame,
# ):
#     """
#     Only keep the prefill latency and batch size of one (performance does not scale with larger batch sizes).
#     """

#     df = df[df['model.stage'] == 'prefill']
#     df = df[df['batch_size'] == 1]

#     # Get all hardware names
#     hardware_names = df['hardware.name'].unique()
#     # Get all model names
#     model_names = df['model.name'].unique()

#     # Create a dictionary to store dict[model][hardware] = latency
#     dict_llm = {}
#     for model_name in model_names:
#         dict_llm[model_name] = {}
#         for hardware_name in hardware_names:
#             # Get the latency for the model and hardware
#             latency_ms = df[(df['model.name'] == model_name) & (df['hardware.name'] == hardware_name)]['time_ms'].values[0]
#             dict_llm[model_name][hardware_name] = latency_ms
    
#     return dict_llm

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

# TODO: + plot prefill latency itself; + plot percentage of retrieval time

def plot_latency(
        # Retrieval
        df_retrieval,
        df_llm,
        model_names=['1B', '3B', '13B'],
        hardware_names=['V100', 'A100', 'H100', 'B100'],
        mode='prefill_latency',
    ):

    assert mode in ['prefill_latency', 'retrieval_latency_perc']

    # plt.style.use('ggplot')
    batch_size = 1
    x_labels = hardware_names
    y_labels = model_names
    x = np.arange(len(x_labels))  # the label locations

    prefill_latency_array = np.zeros((len(model_names), len(hardware_names)))
    retrieval_perc_array = np.zeros((len(model_names), len(hardware_names)))

    # print(f"==== Model {model_name} (rerank top {rerank_top_k}) ====")

    for i, model_name in enumerate(model_names):

        for j, hardware_name in enumerate(hardware_names):

            # Select the appropriate 'data' values for the y-axis based on the batch size
            y_retrieval = df_retrieval[df_retrieval['label'] == 'batch_size={}'.format(batch_size)]['latency'].values
            assert len(y_retrieval) == 1
            y_retrieval = y_retrieval[0] 
            
            y_llm = df_llm[(df_llm['hardware.name'] == hardware_name) & (df_llm['model.name'] == model_name) 
                & (df_llm['model.stage'] == 'prefill') & (df_llm['batch_size'] == batch_size)]['time_ms'].values
            assert len(y_llm) == 1
            y_llm = y_llm[0]

            perc_retrieval = y_retrieval / (y_retrieval + y_llm) * 100
            perc_llm = y_llm / (y_retrieval + y_llm) * 100

            prefill_latency_array[i, j] = y_llm
            retrieval_perc_array[i, j] = perc_retrieval

    # matplot plotlib color map objects: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # cmap = 'RdBu'
    # cmap = 'RdYlGn'
    # cmap = 'coolwarm'
    # cmap = 'bwr'
    # cmap = 'seismic'

    label_font = 12
    tick_font = 12
    legend_font = 12


    fig, ax = plt.subplots(1, 1, figsize=(4,1.2))

    # Heatmap Document: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    if mode == 'prefill_latency':
        array = prefill_latency_array
        cmap = 'summer'
        title = 'LLM prefill latency (ms)'
    elif mode == 'retrieval_latency_perc':
        array = retrieval_perc_array
        cmap = 'winter'
        title = 'Retrieval latency in TTFT (%)'
    
    ax = sns.heatmap(array, cmap=cmap,  annot=True, fmt=".2f", cbar_kws={})

    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Model Size', fontsize=label_font)
    ax.set_xlabel('Inference GPU', fontsize=label_font)
    # ax.set_title('Scores by group and gender')
    # ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels, rotation=0)
    ax.set_title(title, fontsize=label_font)
    


    # number_single_bar(rects1_lower, zero_bottom)
    # number_single_bar(rects1_upper, bar_lower_retrieval)

    # number_stacked_bar(rects1_upper, bar_lower_retrieval)

    # plt.rcParams.update({'figure.autolayout': True})

    for out_type in ['png', 'pdf']:
        plt.savefig(f'./images/llm/rag_{mode}.{out_type}', transparent=False, dpi=200, bbox_inches="tight")


def get_e2e_speedup(
        # Retrieval
        df_retrieval_cpu,
        df_retrieval_fpga, 
        df_llm,
        model_names=['1B', '3B', '13B'],
        hardware_names=['V100', 'A100', 'H100', 'B100'],
    ):

    # print(f"==== Model {model_name} (rerank top {rerank_top_k}) ====")
    speedup_array = np.zeros((len(model_names), len(hardware_names)))
    batch_size = 1    

    for i, model_name in enumerate(model_names):

        for j, hardware_name in enumerate(hardware_names):

            # Select the appropriate 'data' values for the y-axis based on the batch size
            y_retrieval_cpu = df_retrieval_cpu[df_retrieval_cpu['label'] == 'batch_size={}'.format(batch_size)]['latency'].values
            assert len(y_retrieval_cpu) == 1
            y_retrieval_cpu = y_retrieval_cpu[0] 


            y_retrieval_fpga = df_retrieval_fpga[df_retrieval_fpga['label'] == 'batch_size={}'.format(batch_size)]['latency'].values
            assert len(y_retrieval_fpga) == 1
            y_retrieval_fpga = y_retrieval_fpga[0]
            
            y_llm = df_llm[(df_llm['hardware.name'] == hardware_name) & (df_llm['model.name'] == model_name) 
                & (df_llm['model.stage'] == 'prefill') & (df_llm['batch_size'] == batch_size)]['time_ms'].values
            assert len(y_llm) == 1
            y_llm = y_llm[0]

            e2e_baseline = y_llm + y_retrieval_cpu
            e2e_fpga = y_llm + y_retrieval_fpga
            speedup = e2e_baseline / e2e_fpga
            speedup_array[i, j] = speedup

        
    print("E2E Speedup (CPU vs FPGA):\n", speedup_array)
    print(f"E2E Speedup: {np.min(speedup_array):.2f} - {np.max(speedup_array):.2f} x")

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
    model_names=['1B', '3B', '13B']
    hardware_names=['V100', 'A100', 'H100', 'B100']
    		
    df_retrieval_all = get_retrieval_df(
        dataset=dataset, graph_type=graph_type, max_degree=64, ef=64, batch_sizes=[1], CPU_server=CPU_server, add_CPU_network_latency=True)
    df_retrieval_cpu = df_retrieval_all[df_retrieval_all['category'] == 'CPU (Graph)']
    df_retrieval_fpga = df_retrieval_all[df_retrieval_all['category'] == 'Falcon (Best)']
    df_llm = load_llm_latency_df(file_path='./saved_perf_LLM_GPU/llm_perf.csv')

    for mode in ['prefill_latency', 'retrieval_latency_perc']:
            
        plot_latency(
            # Retrieval
            df_retrieval_cpu,
            df_llm,
            model_names=model_names,
            hardware_names=hardware_names,
            mode=mode,
        )

    get_e2e_speedup(
        df_retrieval_cpu,
        df_retrieval_fpga,
        df_llm,
            model_names=model_names,
            hardware_names=hardware_names,
    )
