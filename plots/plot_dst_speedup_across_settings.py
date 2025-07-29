"""
This script shows the speedup brought by DST, given different settings (mostly degrees and dataset dimensionality)

Example usage:
    # plot all (hard-coded) settings: Across-query
    python plot_dst_speedup_across_settings.py --df_path ../perf_test_scripts/saved_df/throughput_FPGA_inter_query_4_chan.pickle --suffix inter_query
    # plot all (hard-coded) settings: intra-query
    python plot_dst_speedup_across_settings.py --df_path ../perf_test_scripts/saved_df/throughput_FPGA_intra_query_4_chan.pickle --suffix intra_query
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd

import seaborn as sns 
# plt.style.use('ggplot')
# plt.style.use('seaborn-pastel') 
# plt.style.use('seaborn-colorblind') 

plt.rcParams['pdf.fonttype'] = 42

sns.set_theme(style="whitegrid")
# sns.set_palette("Paired")
# sns.set_palette("Set2")
sns.set_palette("husl", 3)

parser = argparse.ArgumentParser()

parser.add_argument('--df_path_inter_query', type=str, default="../perf_test_scripts/saved_df/throughput_FPGA_inter_query_4_chan.pickle", help="the performance pickle file to save the dataframe")
parser.add_argument('--df_path_intra_query', type=str, default="../perf_test_scripts/saved_df/throughput_FPGA_intra_query_4_chan.pickle", help="the performance pickle file to save the dataframe")

args = parser.parse_args()


def get_max_speedup(df, graph_type, dataset, max_degree, ef, print_msg=True):
    # select rows 
    df = df.loc[(df['graph_type'] == graph_type) & (df['dataset'] == dataset) & (df['max_degree'] == max_degree) & (df['ef'] == ef)]

    # baseline: max_mg=1, max_mc=1
    row = df.loc[(df['max_cand_per_group'] == 1) & (df['max_group_num_in_pipe'] == 1)]
    assert len(row) == 1
    t_baseline = row['time_ms_kernel'].values[0]

    # get the min time across all settings, get idx
    idx_min = df['time_ms_kernel'].idxmin()
    if print_msg:
        print("Best performance: mc: {}\tmg: {}".format(df.loc[idx_min]['max_cand_per_group'], df.loc[idx_min]['max_group_num_in_pipe']))
    t_min = df.loc[idx_min]['time_ms_kernel']
    speedup = t_baseline / t_min

    return speedup

def get_recall_improvement(df, graph_type, dataset, max_degree, ef, k=10):
    # select rows 
    df = df.loc[(df['graph_type'] == graph_type) & (df['dataset'] == dataset) & (df['max_degree'] == max_degree) & (df['ef'] == ef)]

    # baseline: max_mg=1, max_mc=1
    row = df.loc[(df['max_cand_per_group'] == 1) & (df['max_group_num_in_pipe'] == 1)]
    assert len(row) == 1
    t_baseline = row['time_ms_kernel'].values[0]
    baseline_recall = row['recall_{}'.format(k)].values[0] * 100

    # get the min time across all settings, get idx
    idx_min = df['time_ms_kernel'].idxmin()
    recall = df.loc[idx_min]['recall_{}'.format(k)] * 100
    recall_improvement = recall - baseline_recall 

    return recall_improvement

def plot_speedup(df, graph_type="HNSW", datasets=['SIFT10M', 'Deep10M'], max_degrees=[16, 32, 64], ef=64, suffix='inter_query'):

    # get three subplots, horizontally 
    # fig, ax = plt.subplots(1, 1, figsize=(4, 1.2))
    # for paper, small:
    fig, ax = plt.subplots(1, 1, figsize=(4, 0.8))

    label_font = 13
    markersize = 10
    tick_font = 12
    legend_font = 11

    markers = ['o', '^', 'x', '+', 's', 'D']
    plots = []
    for i, dataset in enumerate(datasets):
        x_labels = [str(md) for md in max_degrees]
        speedup_per_md = [get_max_speedup(df, graph_type, dataset, md, ef, print_msg=False) for md in max_degrees]
        plot = ax.plot(x_labels, speedup_per_md, marker=markers[i], markersize=markersize) #color=color_plot0, 
        plots.append(plot)

    ax.legend([plot[0] for plot in plots], datasets, loc=(0,0), fontsize=legend_font, ncol=2, frameon=False)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=label_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel('max degree per node', fontsize=label_font)
    ax.set_ylabel('DST speedup\nover BFS', fontsize=label_font)
    ax.set_ylim(0, 4)

    # # doted line y = 1, with annotation: BFS
    # ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
    # ax.text(0.0, 1.05, 'Baseline: BFS', fontsize=label_font)
    
    # title with graph name
    if suffix == 'inter_query':
        ax.set_title("Across-query parallel, Graph: " + graph_type, fontsize=label_font)
    elif suffix == 'intra_query':
        ax.set_title("Intra-query parallel, Graph: " + graph_type, fontsize=label_font)

    # print speedup range
    print("\n=== {}, Graph: {} ===".format(suffix, graph_type))
    min_speedup = 100000
    max_speedup = 0
    for dataset in datasets:
        speedup_per_md = [get_max_speedup(df, graph_type, dataset, md, ef, print_msg=False) for md in max_degrees]
        print("Dataset: {}, Speedup: {:.2f}, {:.2f}, {:.2f}".format(dataset, *speedup_per_md))
        min_speedup = min(min_speedup, min(speedup_per_md))
        max_speedup = max(max_speedup, max(speedup_per_md))
    print("Speedup range: {:.2f} - {:.2f}".format(min_speedup, max_speedup))

    min_recall_improve = 100000
    max_recall_improve = 0
    for dataset in datasets:
        recall_improvements = [get_recall_improvement(df, graph_type, dataset, md, ef, k=10) for md in max_degrees]
        print("Dataset: {}, Recall@10 improvement (%): {:.2f}, {:.2f}, {:.2f}".format(dataset, *recall_improvements))
        min_recall_improve = min(min_recall_improve, min(recall_improvements))
        max_recall_improve = max(max_recall_improve, max(recall_improvements))
    print("Recall improvement (%) range: {:.2f} - {:.2f}".format(min_recall_improve, max_recall_improve))

    for out_type in ['png', 'pdf']:
        plt.savefig('./images/dst_speedup_across_settings/dst_speedup_across_settings_{}_{}.{}'.format(graph_type, suffix, out_type), transparent=False, dpi=200, bbox_inches="tight")

    # plt.show()

def plot_speedup_all_subplots(dfs, graph_types=["HNSW", "NSG"], datasets=['SIFT10M', 'Deep10M'], max_degrees=[16, 32, 64], ef=64, suffixs=['inter_query', 'intra_query']):

    assert len(dfs) == len(suffixs)
    n_dfs = len(dfs)
    n_graphs = len(graph_types)

    # get three subplots, horizontally 
    # fig, axs = plt.subplots(n_dfs, n_graphs, figsize=(4 * n_graphs, 2.0 * n_dfs), sharex=True, sharey=True)
    # for paper, small:
    fig, axs = plt.subplots(n_dfs, n_graphs, figsize=(4 * n_graphs, 1.2 * n_dfs), sharex=True, sharey=True)

    # reshape axs into 2D array, in (n_dfs, n_graphs)
    if n_graphs == 1 and n_dfs == 1:
        axs = np.array([[axs]])
    elif n_graphs == 1:
        axs = axs[:, np.newaxis]
    elif n_dfs == 1:
        axs = axs[np.newaxis, :]
    # print axs shape
    # print(axs.shape)

    # vertical paddings
    plt.subplots_adjust(hspace=0.5, wspace=0.15)

    label_font = 13
    markersize = 10
    tick_font = 12
    legend_font = 13

    for dfid, df in enumerate(dfs):
        for gid, graph_type in enumerate(graph_types):
            ax = axs[dfid][gid]
            suffix = suffixs[dfid]
            markers = ['o', '^', 'x', '+', 's', 'D']
            plots = []
            for i, dataset in enumerate(datasets):
                x_labels = [str(md) for md in max_degrees]
                speedup_per_md = [get_max_speedup(df, graph_type, dataset, md, ef, print_msg=False) for md in max_degrees]
                plot = ax.plot(x_labels, speedup_per_md, marker=markers[i], markersize=markersize) #color=color_plot0, 
                plots.append(plot)

            if dfid == 0 and gid == 0:
                ax.legend([plot[0] for plot in plots], datasets, loc=(0.2,1.3), fontsize=legend_font, ncol=3, frameon=False)
            if dfid == n_dfs - 1:
                ax.set_xlabel('max degree per node', fontsize=label_font)
            if gid == 0:
                ax.set_ylabel('Speedup\nover BFS', fontsize=label_font)
                # ax.set_ylabel('DST speedup\nover BFS', fontsize=label_font)

            ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=label_font)
            ax.get_xaxis().set_visible(True)
            ax.set_ylim(0, 4)

            # # doted line y = 1, with annotation: BFS
            # ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
            # ax.text(0.0, 1.05, 'Baseline: BFS', fontsize=label_font)
            
            # title with graph name
            if suffix == 'inter_query':
                ax.set_title("Across-query parallel, Graph: " + graph_type, fontsize=label_font)
            elif suffix == 'intra_query':
                ax.set_title("Intra-query parallel, Graph: " + graph_type, fontsize=label_font)

            # print speedup range
            print("\n=== {}, Graph: {} ===".format(suffix, graph_type))
            min_speedup = 100000
            max_speedup = 0
            for dataset in datasets:
                speedup_per_md = [get_max_speedup(df, graph_type, dataset, md, ef, print_msg=False) for md in max_degrees]
                print("Dataset: {}, Speedup: {:.2f}, {:.2f}, {:.2f}".format(dataset, *speedup_per_md))
                min_speedup = min(min_speedup, min(speedup_per_md))
                max_speedup = max(max_speedup, max(speedup_per_md))
            print("Speedup range: {:.2f} - {:.2f}".format(min_speedup, max_speedup))

            min_recall_improve = 100000
            max_recall_improve = 0
            for dataset in datasets:
                recall_improvements = [get_recall_improvement(df, graph_type, dataset, md, ef, k=10) for md in max_degrees]
                print("Dataset: {}, Recall@10 improvement (%): {:.2f}, {:.2f}, {:.2f}".format(dataset, *recall_improvements))
                min_recall_improve = min(min_recall_improve, min(recall_improvements))
                max_recall_improve = max(max_recall_improve, max(recall_improvements))
            print("Recall improvement (%) range: {:.2f} - {:.2f}".format(min_recall_improve, max_recall_improve))

    for out_type in ['png', 'pdf']:
        plt.savefig('./images/dst_speedup_across_settings/dst_speedup_across_settings_all.{}'.format(out_type), transparent=False, dpi=200, bbox_inches="tight")

    # plt.show()
    #  
if __name__ == "__main__":
    # load dataframe
    df_inter_query = pd.read_pickle(args.df_path_inter_query)
    df_intra_query = pd.read_pickle(args.df_path_intra_query)
    # graph_types = ['HNSW']
    graph_types = ['HNSW', 'NSG']

    datasets = ['SIFT10M', 'Deep10M', "SPACEV10M"]
    # datasets = ['SIFT1M', 'SIFT10M', 'SBERT1M']
    max_degrees=[16, 32, 64]
    ef=64

    # plot individual plots
    # for df, suffix in [(df_inter_query, 'inter_query'), (df_intra_query, 'intra_query')]:
    #     for graph_type in graph_types:
    #         plot_speedup(df, graph_type=graph_type, datasets=datasets, max_degrees=max_degrees, ef=ef, suffix=suffix)

    plot_speedup_all_subplots([df_inter_query, df_intra_query], graph_types=graph_types, datasets=datasets, max_degrees=max_degrees, ef=ef, suffixs=['inter_query', 'intra_query'])
