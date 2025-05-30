
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns 

# plt.style.use('ggplot')
# plt.style.use('seaborn-pastel') 
# plt.style.use('seaborn-colorblind') 

sns.set_theme(style="whitegrid")
# https://seaborn.pydata.org/tutorial/color_palettes.html
# sns.set_palette("Paired")
sns.set_palette("Set2")


parser = argparse.ArgumentParser()
parser.add_argument('--df_path_inter_query', type=str, default="../perf_test_scripts/saved_df/latency_FPGA_inter_query_4_chan.pickle", help="the performance pickle file to save the dataframe")
parser.add_argument('--df_path_intra_query', type=str, default="../perf_test_scripts/saved_df/latency_FPGA_intra_query_4_chan.pickle", help="the performance pickle file to save the dataframe")
args = parser.parse_args()

def get_slowest_fastest_row(df, graph_type, dataset, max_degree, ef, batch_size):
    # select rows 
    df = df.loc[(df['graph_type'] == graph_type) & (df['dataset'] == dataset) & 
                (df['max_degree'] == max_degree) & (df['ef'] == ef) & (df['batch_size'] == batch_size)]

    # baseline: max_mg=1, max_mc=1
    row_baseline = df.loc[(df['max_cand_per_group'] == 1) & (df['max_group_num_in_pipe'] == 1)]
    assert len(row_baseline) == 1
    # t_baseline = row_baseline['time_ms_kernel'].values[0]

    # get the row with min and max time
    row_fastest = df.loc[df['time_ms_kernel'].idxmin()]
    # row_slowest = df.loc[df['time_ms_kernel'].idxmax()]
    # print("Slowest:", row_slowest)
    # print("Fastest:", row_fastest)
    # assert row_slowest['time_ms_kernel'] == t_baseline
    
    return row_baseline, row_fastest

def get_best_qps(df, graph_type, dataset, max_degree, ef, batch_size=10000):
    row_slowest, row_fastest = get_slowest_fastest_row(df, graph_type, dataset, max_degree, ef, batch_size)
    best_qps = 10000 * 1000 / row_fastest['time_ms_kernel']
    return best_qps

def plot_throughput(datasets=["SIFT10M", "Deep10M"], graph_types=["HNSW", "NSG"], max_degree=64, ef=64, CPU_server="r630", GPU_model="V100"):
    
    assert CPU_server in ["r630", "m5.metal", "m5.8xlarge", "sgs-gpu", "gold"]
    
    x_labels = []
    y_cpu = dict()
    y_gpu = dict()
    y_cpu_faiss = dict()
    y_gpu_faiss = dict()
    y_fpga_inter = dict()
    y_fpga_intra = dict()
    recall_1_list = []
    recall_10_list = []
    for dataset in datasets:
        for graph_type in graph_types:
            label = f"{dataset}-{graph_type}".replace("10M", "").replace("1M", "")
            x_labels.append(label)

            batch_size = 10000

            """ CPU Part """
            folder_name_CPU = './saved_perf_CPU'
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
                df_selected = df_cpu[(df_cpu['dataset'] == dataset) & (df_cpu['max_degree'] == max_degree) & (df_cpu['ef'] == ef) & (df_cpu['batch_size'] == batch_size)]
                assert len(df_selected) == 1
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
                df_selected = df_cpu[(df_cpu['dataset'] == dataset) & (df_cpu['max_degree'] == max_degree) & (df_cpu['search_L'] == ef) & (df_cpu['batch_size'] == batch_size)]
                print("CPU:", df_selected)
                assert len(df_selected) == 1
            # get CPU qps
            qps_cpu = df_selected['qps'].values[0]
            y_cpu[label] = qps_cpu
            recall_1_CPU = df_selected['recall_1'].values[0]
            recall_10_CPU = df_selected['recall_10'].values[0]
            recall_1_list.append(recall_1_CPU)
            recall_10_list.append(recall_10_CPU)

            """ CPU Faiss """
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
                recall_10_faiss = df_selected_list[df_selected_list['nprobe'] == nprobe]['recall_10'].values[0]
                if recall_10_faiss >= recall_10_CPU:
                    df_selected = df_selected_list[df_selected_list['nprobe'] == nprobe]
                    break
            assert len(df_selected) == 1

            qps_cpu_faiss = df_selected['qps'].values[0]
            y_cpu_faiss[label] = qps_cpu_faiss

            """ GPU Part """
            folder_name_GPU = './saved_perf_GPU'
            # GPU only supports HNSW
            if graph_type == "HNSW":
                if GPU_model == "3090":
                    f_name_GPU = os.path.join(folder_name_GPU, "perf_df_ggnn_gpu_3090.pickle")
                elif GPU_model == "V100":
                    f_name_GPU = os.path.join(folder_name_GPU, "perf_df_ggnn_gpu_V100.pickle")
                df_gpu = pd.read_pickle(f_name_GPU)
                degree_gpu = 32 # gpu used full knn graph with prunning, thus average degree ~= max degree
                max_iter_gpu = 400 # 400 iterations achieves recall near CPU
                df_selected = df_gpu[(df_gpu['dataset'] == dataset) & (df_gpu['KBuild'] == degree_gpu) & (df_gpu['MaxIter'] == max_iter_gpu) & (df_gpu['batch_size'] == batch_size)]
                print("GPU:", df_selected)
                assert len(df_selected) == 1
                qps_gpu = df_selected['qps'].values[0]
                y_gpu[label] = qps_gpu

            """ GPU Faiss """  
            if GPU_model == "3090":
                df_gpu_faiss = pd.read_pickle(os.path.join(folder_name_GPU, "perf_df_faiss_gpu_3090.pickle"))
            elif GPU_model == "V100":
                df_gpu_faiss = pd.read_pickle(os.path.join(folder_name_GPU, "perf_df_faiss_gpu_V100.pickle"))
            df_selected_list = df_gpu_faiss[(df_gpu_faiss['dataset'] == dataset) & (df_gpu_faiss['batch_size'] == batch_size)]
            nprobe_list_sorted = sorted(df_selected_list['nprobe'].values)
            # use the performance of minimum nprobe that can achieve the same recall as CPU graph (recall_10_CPU)
            df_selected = None
            for nprobe in nprobe_list_sorted:
                recall_10_faiss = df_selected_list[df_selected_list['nprobe'] == nprobe]['recall_10'].values[0]
                if recall_10_faiss >= recall_10_CPU:
                    df_selected = df_selected_list[df_selected_list['nprobe'] == nprobe]
                    break
            assert len(df_selected) == 1

            qps_gpu_faiss = df_selected['qps'].values[0]
            y_gpu_faiss[label] = qps_gpu_faiss

            """ FPGA Part """
            df_fpga_inter = pd.read_pickle(args.df_path_inter_query)
            qps_fpga_inter = get_best_qps(df_fpga_inter, graph_type, dataset, max_degree, ef, batch_size)
            y_fpga_inter[label] = qps_fpga_inter

            df_fpga_intra = pd.read_pickle(args.df_path_intra_query)
            qps_fpga_intra = get_best_qps(df_fpga_intra, graph_type, dataset, max_degree, ef, batch_size)
            y_fpga_intra[label] = qps_fpga_intra
    # print the columns 'dataset', 'recall_10', 'qps'
    print("FPGA throughput (inter):\n", y_fpga_inter)
    print("FPGA throughput (intra):\n", y_fpga_intra)

    def get_error_bar(d):
        """
        Given the key, return a dictionary of std deviation
        """
        dict_error_bar = dict()
        for key in d:
            array = d[key]
            dict_error_bar[key] = np.std(array)
        return dict_error_bar

    def get_mean(d):
        """
        Given the key, return a dictionary of mean
        """
        dict_mean = dict()
        for key in d:
            array = d[key]
            dict_mean[key] = np.average(array)
        return dict_mean

    def get_y_array(d, keys):
        """
        Given a dictionary, and a selection of keys, return an array of y value
        """
        y = []
        for key in keys:
            y.append(d[key])
        return y

    x_labels_with_recall = [x + f"\nR@10≥{recall_10_list[i]*100:.1f}%" for i, x in enumerate(x_labels)]
    x_labels_gpu = [x for x in x_labels if "HNSW" in x]

    if CPU_server == "r630":
        bandwidth_cpu = 240.0
    elif CPU_server == "m5.metal" or CPU_server == "m5.8xlarge":
        bandwidth_cpu = 255.93
    elif CPU_server == "sgs-gpu":
        bandwidth_cpu = 409.6
    elif CPU_server == "gold":
        bandwidth_cpu = 68.256
    if GPU_model == "3090":
        bandwidth_gpu = 938.0 
    elif GPU_model == "V100":
        bandwidth_gpu = 900.0
    bandwidth_fpga = 77.0

    # the qps column is divided by bandwidth
    def normalize_qps(d, bandwidth):
        d_return = dict()
        for k, v in d.items():
            # if v is list
            if isinstance(v, list):
                d_return[k] = [val / bandwidth for val in v]
            else:
                d_return[k] = v / bandwidth
        return d_return
    
    y_cpu_norm = normalize_qps(y_cpu, bandwidth_cpu)
    y_cpu_faiss_norm = normalize_qps(y_cpu_faiss, bandwidth_cpu)
    y_gpu_norm = normalize_qps(y_gpu, bandwidth_gpu)
    y_gpu_faiss_norm = normalize_qps(y_gpu_faiss, bandwidth_gpu)
    y_fpga_inter_norm = normalize_qps(y_fpga_inter, bandwidth_fpga)
    y_fpga_intra_norm = normalize_qps(y_fpga_intra, bandwidth_fpga)

    y_cpu_means = get_y_array(get_mean(y_cpu), x_labels)
    y_cpu_faiss_means = get_y_array(get_mean(y_cpu_faiss), x_labels)
    y_gpu_means = get_y_array(get_mean(y_gpu), x_labels_gpu)
    y_gpu_faiss_means = get_y_array(get_mean(y_gpu_faiss), x_labels)
    y_fpga_inter_means = get_y_array(get_mean(y_fpga_inter), x_labels)
    y_fpga_intra_means = get_y_array(get_mean(y_fpga_intra), x_labels)

    # print("CPU throughput:", y_cpu_means)
    # print("GPU throughput:", y_gpu_means)
    # print("FPGA throughput:", y_fpga_inter_means)
    speedup_over_cpu = np.array(y_fpga_inter_means) / np.array(y_cpu_means)
    speedup_over_cpu_faiss = np.array(y_fpga_inter_means) / np.array(y_cpu_faiss_means)
    y_fpga_inter_means_hnsw = [y_fpga_inter_means[i] for i, x in enumerate(x_labels) if "HNSW" in x]
    y_fpga_intra_means_hnsw = [y_fpga_intra_means[i] for i, x in enumerate(x_labels) if "HNSW" in x]
    speedup_over_gpu = np.array(y_fpga_inter_means_hnsw) / np.array(y_gpu_means)
    speedup_over_gpu_faiss = np.array(y_fpga_inter_means) / np.array(y_gpu_faiss_means)
    print("FPGA throughput speedup over CPU: {:.2f}~{:.2f}".format(np.min(speedup_over_cpu), np.max(speedup_over_cpu)))
    print("FPGA throughput speedup over GPU: {:.2f}~{:.2f}".format(np.min(speedup_over_gpu), np.max(speedup_over_gpu)))
    print("FPGA throughput speedup over CPU Faiss: {:.2f}~{:.2f}".format(np.min(speedup_over_cpu_faiss), np.max(speedup_over_cpu_faiss)))
    print("FPGA throughput speedup over GPU Faiss: {:.2f}~{:.2f}".format(np.min(speedup_over_gpu_faiss), np.max(speedup_over_gpu_faiss)))

    y_cpu_means_norm = get_y_array(get_mean(y_cpu_norm), x_labels)
    y_cpu_faiss_means_norm = get_y_array(get_mean(y_cpu_faiss_norm), x_labels)
    y_gpu_means_norm = get_y_array(get_mean(y_gpu_norm), x_labels_gpu)
    y_gpu_faiss_means_norm = get_y_array(get_mean(y_gpu_faiss_norm), x_labels)
    y_fpga_inter_means_norm = get_y_array(get_mean(y_fpga_inter_norm), x_labels)
    y_fpga_intra_means_norm = get_y_array(get_mean(y_fpga_intra_norm), x_labels)
    
    # print("Normalized CPU throughput:", y_cpu_means_norm)
    # print("Normalized GPU throughput:", y_gpu_means_norm)
    # print("Normalized FPGA throughput:", y_fpga_inter_means_norm)
    speedup_over_cpu_norm = np.array(y_fpga_inter_means_norm) / np.array(y_cpu_means_norm)
    speedup_over_cpu_faiss_norm = np.array(y_fpga_inter_means_norm) / np.array(y_cpu_faiss_means_norm)
    y_fpga_inter_means_norm_hnsw = [y_fpga_inter_means_norm[i] for i, x in enumerate(x_labels) if "HNSW" in x]
    y_fpga_intra_means_norm_hnsw = [y_fpga_intra_means_norm[i] for i, x in enumerate(x_labels) if "HNSW" in x]
    speedup_over_gpu_norm = np.array(y_fpga_inter_means_norm_hnsw) / np.array(y_gpu_means_norm)
    speedup_over_gpu_faiss_norm = np.array(y_fpga_inter_means_norm) / np.array(y_gpu_faiss_means_norm)
    print("Normalized FPGA throughput speedup over CPU: {:.2f}~{:.2f}".format(np.min(speedup_over_cpu_norm), np.max(speedup_over_cpu_norm)))
    print("Normalized FPGA throughput speedup over GPU: {:.2f}~{:.2f}".format(np.min(speedup_over_gpu_norm), np.max(speedup_over_gpu_norm)))
    print("Normalized FPGA throughput speedup over CPU Faiss: {:.2f}~{:.2f}".format(np.min(speedup_over_cpu_faiss_norm), np.max(speedup_over_cpu_faiss_norm)))
    print("Normalized FPGA throughput speedup over GPU Faiss: {:.2f}~{:.2f}".format(np.min(speedup_over_gpu_faiss_norm), np.max(speedup_over_gpu_faiss_norm)))

    y_cpu_error_bar = get_y_array(get_error_bar(y_cpu), x_labels)
    y_cpu_faiss_error_bar = get_y_array(get_error_bar(y_cpu_faiss), x_labels)
    y_gpu_error_bar = get_y_array(get_error_bar(y_gpu), x_labels_gpu)
    y_gpu_faiss_error_bar = get_y_array(get_error_bar(y_gpu_faiss), x_labels)
    y_fpga_inter_error_bar = get_y_array(get_error_bar(y_fpga_inter), x_labels)
    y_fpga_intra_error_bar = get_y_array(get_error_bar(y_fpga_intra), x_labels)

    y_cpu_error_bar_norm = get_y_array(get_error_bar(y_cpu_norm), x_labels)
    y_cpu_faiss_error_bar_norm = get_y_array(get_error_bar(y_cpu_faiss_norm), x_labels)
    y_gpu_error_bar_norm = get_y_array(get_error_bar(y_gpu_norm), x_labels_gpu)
    y_gpu_faiss_error_bar_norm = get_y_array(get_error_bar(y_gpu_faiss_norm), x_labels)
    y_fpga_inter_error_bar_norm = get_y_array(get_error_bar(y_fpga_inter_norm), x_labels)
    y_fpga_intra_error_bar_norm = get_y_array(get_error_bar(y_fpga_intra_norm), x_labels)

    x = np.arange(len(x_labels))  # the label locations
    x_gpu = np.array([x_labels.index(i) for i in x_labels_gpu if "HNSW" in i])
    # x except x_gpu
    x_gpu_na = np.array([x_labels.index(i) for i in x_labels if "HNSW" not in i])
    width = 0.1  # the width of the bars

    # two subplots (top and down), share the x axis
    fig, (ax, ax_norm) = plt.subplots(2, 1, figsize=(18, 3))
    fig.subplots_adjust(hspace=0.1)
    ax_norm.get_shared_x_axes().join(ax, ax_norm)

    label_font = 12
    tick_font = 11
    legend_font = 12
    na_font = 10
    title_font = 12
    error_bar_cap_size = 3

    # hatch styles: https://matplotlib.org/3.4.3/gallery/shapes_and_collections/hatch_style_reference.html
    # hatch_fpga = '---' 
    # hatch_gpu = '////'
    x_fpga_inter_position = x - 3 * width
    x_fpga_intra_position = x - 2 * width
    x_cpu_position = x - width
    x_cpu_faiss_position = x
    x_gpu_position = x_gpu + width
    x_gpu_faiss_position = x + 2 * width
    
    rects_fpga_inter = ax.bar(x_fpga_inter_position, y_fpga_inter_means, width)#, hatch=hatch_fpga)
    rects_fpga_intra = ax.bar(x_fpga_intra_position, y_fpga_intra_means, width)#, hatch=hatch_fpga)
    rects_cpu  = ax.bar(x_cpu_position, y_cpu_means, width)#, hatch='//')
    rects_cpu_faiss  = ax.bar(x_cpu_faiss_position, y_cpu_faiss_means, width)#, hatch='//')
    rects_gpu = ax.bar(x_gpu_position, y_gpu_means, width)#, hatch=hatch_gpu)
    rects_gpu_faiss = ax.bar(x_gpu_faiss_position, y_gpu_faiss_means, width)#, hatch=hatch_gpu)
    ax.errorbar(x_fpga_inter_position , y_fpga_inter_means, yerr=y_fpga_inter_error_bar, fmt=',', ecolor='black', capsize=error_bar_cap_size)
    ax.errorbar(x_fpga_intra_position , y_fpga_intra_means, yerr=y_fpga_intra_error_bar, fmt=',', ecolor='black', capsize=error_bar_cap_size)
    ax.errorbar(x_cpu_position, y_cpu_means, yerr=y_cpu_error_bar, fmt=',', ecolor='black', capsize=error_bar_cap_size)
    ax.errorbar(x_cpu_faiss_position, y_cpu_faiss_means, yerr=y_cpu_faiss_error_bar, fmt=',', ecolor='black', capsize=error_bar_cap_size)
    ax.errorbar(x_gpu_position, y_gpu_means, yerr=y_gpu_error_bar, fmt=',', ecolor='black', capsize=error_bar_cap_size)
    ax.errorbar(x_gpu_faiss_position, y_gpu_faiss_means, yerr=y_gpu_faiss_error_bar, fmt=',', ecolor='black', capsize=error_bar_cap_size)

    rects_fpga_inter_norm = ax_norm.bar(x_fpga_inter_position, y_fpga_inter_means_norm, width)#, hatch=hatch_fpga)
    rects_fpga_intra_norm = ax_norm.bar(x_fpga_intra_position, y_fpga_intra_means_norm, width)#, hatch=hatch_fpga)
    rects_cpu_norm  = ax_norm.bar(x_cpu_position, y_cpu_means_norm, width)
    rects_cpu_faiss_norm  = ax_norm.bar(x_cpu_faiss_position , y_cpu_faiss_means_norm, width)
    rects_gpu_norm = ax_norm.bar(x_gpu_position, y_gpu_means_norm, width)#, hatch=hatch_gpu)
    rects_gpu_faiss_norm = ax_norm.bar(x_gpu_faiss_position, y_gpu_faiss_means_norm, width)#, hatch=hatch_gpu)
    ax_norm.errorbar(x_fpga_inter_position , y_fpga_inter_means_norm, yerr=y_fpga_inter_error_bar_norm, fmt=',', ecolor='black', capsize=error_bar_cap_size)
    ax_norm.errorbar(x_fpga_intra_position , y_fpga_intra_means_norm, yerr=y_fpga_intra_error_bar_norm, fmt=',', ecolor='black', capsize=error_bar_cap_size)
    ax_norm.errorbar(x_cpu_position, y_cpu_means_norm, yerr=y_cpu_error_bar_norm, fmt=',', ecolor='black', capsize=error_bar_cap_size)
    ax_norm.errorbar(x_cpu_faiss_position, y_cpu_faiss_means_norm, yerr=y_cpu_faiss_error_bar_norm, fmt=',', ecolor='black', capsize=error_bar_cap_size)
    ax_norm.errorbar(x_gpu_position, y_gpu_means_norm, yerr=y_gpu_error_bar_norm, fmt=',', ecolor='black', capsize=error_bar_cap_size)
    ax_norm.errorbar(x_gpu_faiss_position, y_gpu_faiss_means_norm, yerr=y_gpu_faiss_error_bar_norm, fmt=',', ecolor='black', capsize=error_bar_cap_size)


    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('QPS without SLA', fontsize=label_font)
    ax.set_ylabel('QPS', fontsize=label_font)
    ax_norm.set_ylabel('QPS per GB/s', fontsize=label_font)
    # ax_norm.set_ylabel('Norm. QPS\n(QPS per GB/s)', fontsize=label_font)
    # ax_norm.set_ylabel('QPS per GB/s', fontsize=label_font)
    # ax.set_title('Throughput comparison between CPU and FPGA')
    # ax.set_xticks(x) 
    # no x ticks for ax
    ax.set_xticks([])
    ax_norm.set_xticks(x)
    ax_norm.set_xticklabels(x_labels_with_recall, rotation=0, fontsize=tick_font)
    ax.legend([rects_fpga_inter, rects_fpga_intra, rects_cpu, rects_cpu_faiss, rects_gpu, rects_gpu_faiss], 
              [ "Falcon (Across-query)", "Falcon (Intra-query)", "CPU (Graph)", "CPU (IVF)", "GPU (Graph)", "GPU (IVF)", ], loc=(0.1, 1.02), ncol=6, \
        facecolor='white', framealpha=1, frameon=False, fontsize=legend_font)
    # ax.legend([rects_fpga_inter, rects_fpga_intra, rects_cpu, rects_cpu_faiss, rects_gpu, rects_gpu_faiss], 
    #           [ "Falcon (Across-query)", "Falcon (Intra-query)", "CPU (Graph)", "CPU (IVF)", "GPU (Graph)", "GPU (IVF)", ], loc=(0.2, 1.0), ncol=3, \
    #     facecolor='white', framealpha=1, frameon=False, fontsize=legend_font)

    def autolabel(rects, ax=ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate("{:.2E}".format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90)

    # autolabel(rects_cpu, ax)
    # autolabel(rects_fpga_inter, ax)
    # autolabel(rects_gpu, ax)

    # autolabel(rects_cpu_norm, ax_norm)
    # autolabel(rects_fpga_inter_norm, ax_norm)
    # autolabel(rects_gpu_norm, ax_norm)

    # log scale
    ax.set_yscale('log')
    ax.set_ylim(1e2, 1e6)
    # for positions of x_gpu_na, write N/A vertically
    for x_na in x_gpu_na:
        ax.text(x_na + width, 1.2 * 1e2, 'N/A', ha='center', va='bottom', rotation=90, fontsize=na_font)

    ax_norm.set_yscale('log')
    ax_norm.set_ylim(1, 8e2)
    # for positions of x_gpu_na, write N/A vertically
    for x_na in x_gpu_na:
        ax_norm.text(x_na + width, 1.2 * 1, 'N/A', ha='center', va='bottom', rotation=90, fontsize=na_font)
    
    # remove vertical grids
    ax.grid(axis='x')
    ax_norm.grid(axis='x')

    # plt.rcParams.update({'figure.autolayout': True})

    for out_type in ['png', 'pdf']:
        plt.savefig(f'./images/throughput_CPU_GPU_FPGA.{out_type}', transparent=False, dpi=200, bbox_inches="tight")
    # plt.show()

if __name__ == "__main__":
    # CPU_server = "gold"
    CPU_server = "m5.metal"
    # CPU_server = "m5.8xlarge"
    # CPU_server = "r630"
    # CPU_server = "sgs-gpu"
    datasets=["SIFT10M", "Deep10M", "SPACEV10M"]
    # graph_types=["HNSW"]
    graph_types=["HNSW", "NSG"]
    max_degree=64
    ef=64

    plot_throughput(datasets=datasets, graph_types=graph_types, max_degree=max_degree, ef=ef, CPU_server=CPU_server)