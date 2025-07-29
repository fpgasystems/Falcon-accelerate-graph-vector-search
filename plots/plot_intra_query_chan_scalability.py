"""
This script shows the scalability of intra-query parallelism across different number of channels

Example usage:

    python plot_intra_query_chan_scalability.py \
        --df_path_1_chan ../perf_test_scripts/saved_df/latency_FPGA_intra_query_1_chan.pickle \
        --df_path_2_chan ../perf_test_scripts/saved_df/latency_FPGA_intra_query_2_chan.pickle \
        --df_path_4_chan ../perf_test_scripts/saved_df/latency_FPGA_intra_query_4_chan.pickle
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

# sns.set_theme(style="whitegrid")
# plt.style.use('seaborn')
# plt.style.use('seaborn-colorblind')
plt.style.use('seaborn-pastel')

plt.rcParams['pdf.fonttype'] = 42

parser = argparse.ArgumentParser()

parser.add_argument('--df_path_1_chan', type=str, default="perf_df_1_chan.pickle", help="the performance pickle file to save the dataframe")
parser.add_argument('--df_path_2_chan', type=str, default="perf_df_2_chan.pickle", help="the performance pickle file to save the dataframe")
parser.add_argument('--df_path_4_chan', type=str, default="perf_df_4_chan.pickle", help="the performance pickle file to save the dataframe")

args = parser.parse_args()


def get_slowest_fastest_row(df, graph_type, dataset, max_degree, ef, batch_size=1):
    # select rows 
    df = df.loc[(df['graph_type'] == graph_type) & (df['dataset'] == dataset) & 
                (df['max_degree'] == max_degree) & (df['ef'] == ef) & (df['batch_size'] == batch_size)]

    # baseline: max_mg=1, max_mc=1
    row_baseline = df.loc[(df['max_cand_per_group'] == 1) & (df['max_group_num_in_pipe'] == 1)]
    assert len(row_baseline) == 1
    t_baseline = row_baseline['time_ms_kernel'].values[0]

    # get the row with min and max time
    row_fastest = df.loc[df['time_ms_kernel'].idxmin()]
    row_slowest = df.loc[df['time_ms_kernel'].idxmax()]
    assert row_slowest['time_ms_kernel'] == t_baseline
    
    return row_slowest, row_fastest

def plot_speedup(df_1_chan, df_2_chan, df_4_chan, graph_type="HNSW", dataset="SIFT1M", max_degree=64, ef=64):

    df_chan_list = [df_1_chan, df_2_chan, df_4_chan]
    # get three subplots, horizontally 
    fig, ax = plt.subplots(1, 1, figsize=(3, 1.8))
    x = ["1", "2", "4"]

    label_font = 12
    markersize = 8
    tick_font = 10
    legend_font = 12

    # 2 x 2 plots
    # plot group 1: speed up based on time for slowest / fastest run for different PE numbers - with solid lines (y axis 1)
    # plot group 2: ratio of avg_hops for slowest / fastest run for different PE numbers - with dashed lines (y axis 2)
    row_baseline_slowest_1_chan, _ = get_slowest_fastest_row(df_1_chan, graph_type, dataset, max_degree, ef)

    # group 1 
    y_speedup_dst = []
    y_speedup_bfs = []
    # group 2
    y_avg_hops_ratio_dst = []
    y_avg_hops_ratio_bfs = []
    # get the slowest and fastest row for each number of PEs, and compute speedup to baselines
    for df in df_chan_list:
        row_slowest, row_fastest = get_slowest_fastest_row(df, graph_type, dataset, max_degree, ef)
        
        y_speedup_bfs.append(row_baseline_slowest_1_chan['time_ms_kernel'] / row_slowest['time_ms_kernel'])
        y_speedup_dst.append(row_baseline_slowest_1_chan['time_ms_kernel'] / row_fastest['time_ms_kernel'])

        y_avg_hops_ratio_bfs.append(row_slowest['avg_hops'] / row_baseline_slowest_1_chan['avg_hops'])
        y_avg_hops_ratio_dst.append(row_fastest['avg_hops'] / row_baseline_slowest_1_chan['avg_hops'])

    # print info, with :.2f precision
    print("\n=== Dataset: {}, Graph: {} ===".format(dataset, graph_type))
    print("DST Speedup over 1-BFC BFS: {:.2f}, {:.2f}, {:.2f}".format(*y_speedup_dst))
    print("BFS Speedup over 1-BFC BFS: {:.2f}, {:.2f}, {:.2f}".format(*y_speedup_bfs))
    print("DST Speedup over BFS (1, 2, 4 chan): {:.2f}, {:.2f}, {:.2f}".format(*y_speedup_dst / np.array(y_speedup_bfs)))
    print("DST Speedup over 1-BFC DST: {:.2f}, {:.2f}, {:.2f}".format(*y_speedup_dst / y_speedup_dst[0]))

    print("DST Hops Ratio to 1-BFC BFS: {:.2f}, {:.2f}, {:.2f}".format(*y_avg_hops_ratio_dst))

    # plot 
    plt_speedup_dst = ax.plot(x, y_speedup_dst, marker='o', markersize=markersize, label='DST speedup')
    plt_speedup_bfs = ax.plot(x, y_speedup_bfs, marker='o', markersize=markersize, label='BFS speedup')
    ax.set_xlabel('#Bloom-fetch-compute (BFC) unit', fontsize=label_font)
    ax.set_ylabel('Norm. speedup\nover 1-BFC BFS', fontsize=label_font)
    # show graph type and dataset on upper left
    ax.text(0.1, y_speedup_dst[-1], '{},{}'.format(dataset, graph_type), fontsize=label_font, horizontalalignment='left', verticalalignment='top')
    
    # plot group 2 using dashed lines
    ax2 = ax.twinx()
    plt_avg_hops_ratio_dst = ax2.plot(x, y_avg_hops_ratio_dst, marker='^', markersize=markersize, label='DST hops', linestyle='--')
    plt_avg_hops_ratio_bfs = ax2.plot(x, y_avg_hops_ratio_bfs, marker='^', markersize=markersize, label='BFS hops', linestyle='--')
    ax2.set_ylabel('Norm. #hops\nover 1-BFC BFS', fontsize=label_font)

    # mark the speedup of 4 chan dst to 1 chan dst, with a vertical line (with speedup), with y range from  y_speedup_dst[0] to  y_speedup_dst[-1]
    # dst_speedup_4_to_1 = y_speedup_dst[-1] / y_speedup_dst[0]
    # ax.arrow(2, y_speedup_dst[0], 0, y_speedup_dst[-1] - y_speedup_dst[0], head_width=0.1, head_length=0.1, fc='gray', ec='gray', shape='full')
    # ax.arrow(2, y_speedup_dst[-1], 0, y_speedup_dst[0] - y_speedup_dst[-1], head_width=0.1, head_length=0.1, fc='gray', ec='gray', shape='full')
    # ax.annotate("", xy=(2, y_speedup_dst[-1]), xytext=(2, y_speedup_dst[0]), arrowprops=dict(arrowstyle="<->", color='gray'))
    # ax.axhline(y=y_speedup_dst[0], xmin=0.05, xmax=0.95, color='gray', linestyle=':')
    # ax.text(1.95, (y_speedup_dst[0] + y_speedup_dst[-1]) / 2, "{:.2f}x speedup".format(dst_speedup_4_to_1), 
    #         fontsize=label_font, horizontalalignment='right', verticalalignment='center', rotation=90)

    # mark speedup of DST over BFS given different PEs
    for i in range(len(x)):
        ax.annotate("", xy=(i, y_speedup_dst[i]), xytext=(i, y_speedup_bfs[i]), arrowprops=dict(arrowstyle="<->", color='gray'))
        ax.text(i, (y_speedup_dst[i] + y_speedup_bfs[i]) / 2, "{:.2f}x".format(y_speedup_dst[i] / y_speedup_bfs[i]), 
                fontsize=label_font, horizontalalignment='right', verticalalignment='center', rotation=90)


    # set legend
    plots = plt_speedup_dst + plt_speedup_bfs + plt_avg_hops_ratio_dst + plt_avg_hops_ratio_bfs
    ax.legend(plots, [p.get_label() for p in plots], loc=(-0.2, 1.05), fontsize=legend_font, ncol=2, frameon=False)

	# add horizontal grid
    # ax.grid(axis='y', linestyle='-', alpha=0.5, linewidth=1)

    # set x lim
    ax.set_xlim(-0.2, 2.1)

    for out_type in ['png', 'pdf']:
        plt.savefig('./images/intra_query_chan_scalability/intra_query_scalability_{}_{}.{}'.format(graph_type, dataset, out_type), transparent=False, dpi=200, bbox_inches="tight")

    # plt.show()
 
if __name__ == "__main__":
    # load dataframe
    df_1_chan = pd.read_pickle(args.df_path_1_chan)
    df_2_chan = pd.read_pickle(args.df_path_2_chan)
    df_4_chan = pd.read_pickle(args.df_path_4_chan)
    # graph_types = ['HNSW']
    datasets = ['SIFT10M', 'Deep10M', 'SPACEV10M']
    graph_types = ['HNSW', 'NSG']

    for graph_type in graph_types:
        for dataset in datasets:
            plot_speedup(df_1_chan, df_2_chan, df_4_chan, graph_type, dataset)
