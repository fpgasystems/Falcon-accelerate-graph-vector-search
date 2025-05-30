"""
This script shows cross-over latency between intra-query and inter-query parallelism.

Example usage:

    python plot_intra_vs_inter_performance.py --df_path_intra_query ../perf_test_scripts/saved_df/latency_FPGA_intra_query_4_chan.pickle --df_path_inter_query ../perf_test_scripts/saved_df/latency_FPGA_inter_query_4_chan.pickle
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd

# plt.style.use('seaborn')
plt.style.use('seaborn-colorblind')

parser = argparse.ArgumentParser()

parser.add_argument('--df_path_intra_query', type=str, default="perf_df_intra.pickle", help="the performance pickle file to save the dataframe")
parser.add_argument('--df_path_inter_query', type=str, default="perf_df_inter.pickle", help="the performance pickle file to save the dataframe")

args = parser.parse_args()


def get_slowest_fastest_row(df, graph_type, dataset, max_degree, ef, batch_size):
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

def get_best_latency(df, graph_type, dataset, max_degree, ef, batch_size):
    row_slowest, row_fastest = get_slowest_fastest_row(df, graph_type, dataset, max_degree, ef, batch_size)
    best_latency = row_fastest['avg_latency_per_batch_ms']
    return best_latency

def plot_latency(df_intra, df_inter, graph_type="HNSW", dataset="SIFT1M", max_degree=64, ef=64):
    """
    plot latency over different batch sizes 
    """
    # get three subplots, horizontally 
    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    
    batch_sizes = [1, 2, 4, 8]
    x = [str(i) for i in batch_sizes]

    label_font = 10
    markersize = 8
    tick_font = 8
    legend_font = 9

    y_latency_intra = []
    y_latency_inter = []
    for batch_size in batch_sizes:
        y_latency_intra.append(get_best_latency(df_intra, graph_type, dataset, max_degree, ef, batch_size))
        y_latency_inter.append(get_best_latency(df_inter, graph_type, dataset, max_degree, ef, batch_size))
        
    # plot 
    plt_latency_intra = ax.plot(x, y_latency_intra, marker='o', markersize=markersize, label='Intra-query parallel')
    plt_latency_inter = ax.plot(x, y_latency_inter, marker='x', markersize=markersize, label='Inter-query parallel')
    plots = plt_latency_intra + plt_latency_inter
    ax.set_xlabel('Batch size', fontsize=label_font)
    ax.set_ylabel('Average Latency (ms)', fontsize=label_font)
    # show graph type and dataset on upper left
    ax.text(0.1, (y_latency_intra[0] + y_latency_intra[-1]) / 2, '{},{}'.format(dataset, graph_type), fontsize=label_font, horizontalalignment='left', verticalalignment='top')
    # set legend
    ax.legend(plots, [p.get_label() for p in plots], loc="upper left", fontsize=legend_font, ncol=1)
    # set tick font size
    ax.tick_params(axis='both', which='major', labelsize=tick_font)


    # # print info, with :.2f precision
    print("\n=== Dataset: {}, Graph: {} ===".format(dataset, graph_type))
    print("Intra-query Speedup over Inter-query: ")
    for i in range(len(batch_sizes)):
        print("Batch size {}: {:.2f}".format(batch_sizes[i], y_latency_inter[i] / y_latency_intra[i]))
    # print("BFS Speedup over 1-PP BFS: {:.2f}, {:.2f}, {:.2f}".format(*y_speedup_bfs))
    # print("DST Speedup over BFS (1, 2, 4 chan): {:.2f}, {:.2f}, {:.2f}".format(*y_speedup_dst / np.array(y_speedup_bfs)))
    # print("DST Speedup over 1-PP DST: {:.2f}, {:.2f}, {:.2f}".format(*y_speedup_dst / y_speedup_dst[0]))

    # print("DST Hops Ratio to 1-PP BFS: {:.2f}, {:.2f}, {:.2f}".format(*y_avg_hops_ratio_dst))

    plt.savefig('./images/intra_vs_inter_performance/intra_inter_query_crossover_{}_{}.png'.format(graph_type, dataset), transparent=False, dpi=200, bbox_inches="tight")

    # plt.show()
 
if __name__ == "__main__":
    # load dataframe
    df_intra = pd.read_pickle(args.df_path_intra_query)
    df_inter = pd.read_pickle(args.df_path_inter_query)
    graph_types = ['HNSW']
    datasets = ['SIFT1M', 'SIFT10M', 'SBERT1M']
    # graph_types = ['HNSW', 'NSG']

    for graph_type in graph_types:
        for dataset in datasets:
            plot_latency(df_intra, df_inter, graph_type, dataset)
