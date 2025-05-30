"""
This script visualizes the heatmap of the speedup, nodes, and recall on FPGA using DST.

Example usage:

    # plot a specific setting
    python plot_dst_speedup_nodes_recall.py --graph_type HNSW --dataset SIFT1M --max_degree 64 --ef 64 --df_path ../perf_test_scripts/saved_df/throughput_FPGA_inter_query_4_chan.pickle --suffix inter_query

    # plot all (hard-coded) settings: intra-query
    python plot_dst_speedup_nodes_recall.py --plot_all 1 --df_path ../perf_test_scripts/saved_df/throughput_FPGA_intra_query_4_chan.pickle --suffix intra_query --max_mc 4 --max_mg 7
    # plot all (hard-coded) settings: Across-query
    python plot_dst_speedup_nodes_recall.py --plot_all 1 --df_path ../perf_test_scripts/saved_df/throughput_FPGA_inter_query_4_chan.pickle --suffix inter_query --max_mc 4 --max_mg 7
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--plot_all', type=int, default=0, help="Ignore the following parameters and loop over a set of configurations")
parser.add_argument('--graph_type', type=str, default="HNSW", help="HNSW or NSG")
parser.add_argument('--dataset', type=str, default="SIFT1M", help="SIFT1M, SIFT10M, SBERT1M, etc.")
parser.add_argument('--max_degree', type=int, default=64, help="max degree of the graph (base layer for HNSW)")
parser.add_argument('--ef', type=int, default=64, help="max result queue size")
parser.add_argument('--df_path', type=str, default="perf_df.pickle", help="the performance pickle file to save the dataframe")
parser.add_argument('--suffix', type=str, default="inter_query", help="suffix of the output")
parser.add_argument('--max_mc', type=int, default=3, help="max mc")
parser.add_argument('--max_mg', type=int, default=6, help="max mg")

args = parser.parse_args()


def plot_heatmap(df, graph_type, dataset, max_degree, ef, suffix='inter_query', max_mg=6, max_mc=3, show_title=True):

    # select rows 
    df = df.loc[(df['graph_type'] == graph_type) & (df['dataset'] == dataset) & (df['max_degree'] == max_degree) & 
             (df['ef'] == ef)]

    # get the speedup, nodes, recall
    time_array = np.zeros((max_mc, max_mg))
    nodes_array = np.zeros((max_mc, max_mg))
    recall_array = np.zeros((max_mc, max_mg))
    for mc in range(max_mc):
        for mg in range(max_mg):
            row = df.loc[(df['max_cand_per_group'] == 1 + mc) & (df['max_group_num_in_pipe'] == 1 + mg)]
            print("row: ", row)
            assert len(row) == 1
            time_array[mc][mg] = row['time_ms_kernel'].values[0]
            nodes_array[mc][mg] = row['avg_hops'].values[0]
            recall_array[mc][mg] = row['recall_10'].values[0] * 100

    # speedup over max_mg=1, max_mc=1
    speedup_array = np.zeros((max_mc, max_mg))
    for mc in range(max_mc):
        for mg in range(max_mg):
            speedup_array[mc][mg] = time_array[0][0] / time_array[mc][mg]
    normalized_node_array = nodes_array / nodes_array[0][0]

    # Compute utilization == #Nodes / Latency ~ Nodes x Speedup
    compute_util_array = normalized_node_array * speedup_array


    print("Time (ms) array:\n", time_array)
    print("Nodes array:\n", nodes_array)
    print("Recall array:\n", recall_array)
    print("Speedup array:\n", speedup_array)

    def plot_all():
        # get three subplots, horizontally 
        fig, ax_array = plt.subplots(1, 3, figsize=(18, 1.8))
        (ax_speedup, ax_nodes, ax_recall) = ax_array
        # set space between subplots
        plt.subplots_adjust(wspace=0.1)

        # matplotlib color map objects: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        cmap = 'RdBu'
        # cmap = 'RdYlGn'
        # cmap = 'coolwarm'
        # cmap = 'bwr'
        # cmap = 'seismic'


        label_font = 14
        tick_font = 12
        legend_font = 12

        # Heatmap Document: https://seaborn.pydata.org/generated/seaborn.heatmap.html
        ax_speedup = seaborn.heatmap(speedup_array, cmap='YlGn', ax=ax_speedup, annot=True, fmt=".2f", cbar_kws={})
        ax_nodes = seaborn.heatmap(normalized_node_array, cmap='OrRd', ax=ax_nodes, annot=True, fmt=".2f", cbar_kws={})
        ax_recall = seaborn.heatmap(recall_array, cmap='Blues', ax=ax_recall,annot=True, fmt=".2f", cbar_kws={})
        # ax_heatmap = seaborn.heatmap(data, cmap=cmap, cbar_kws={'label': 'colorbar title'})
        # set colorbar label size by hacking: https://stackoverflow.com/questions/48586738/seaborn-heatmap-colorbar-label-font-size
        ax_speedup.figure.axes[-1].yaxis.label.set_size(12)

        x_tick_labels = np.arange(1, max_mg + 1)
        y_tick_labels = np.arange(1, max_mc + 1)
        # 2**10 is on the very top in the heatmap, we want to put it down
        # y_tick_labels.reverse()


        for ax in ax_array:
            ax.set_xticklabels(x_tick_labels)
            ax.set_yticklabels(y_tick_labels)
            ax.set_xlabel('max #groups in the pipeline (mg)', fontsize=label_font, labelpad=0)
            ax.set_ylabel('max #cand.\nper group (mc)', fontsize=label_font, labelpad=5)
            ax.tick_params(length=0, top=False, bottom=False, left=False, right=False, 
                labelleft=True, labelbottom=True)
        plt.yticks(rotation=0)
        # ax_speedup.ticklabel_format(axis='both', style='sci')
        # ax_speedup.set_yscale("log")


        # on the upper right corner, mark the dataset name
        # ax_recall.text(0.75, 1.12, f"{dataset}, {graph_type}", horizontalalignment='left', verticalalignment='center', transform=ax_recall.transAxes, fontsize=legend_font)

        # add plot title
        if suffix == "inter_query":
            plt.suptitle(f'(a) Across-query parallelism, {graph_type} on {dataset}, max_degree={max_degree}', fontsize=14, y=1.2)
            if show_title:
                ax_speedup.set_title('Speedup over BFS', fontsize=label_font)
                ax_nodes.set_title('Normalized average #hops per search', fontsize=label_font)
                ax_recall.set_title('Recall R@10', fontsize=label_font)
        elif suffix == "intra_query":
            plt.suptitle(f'(b) Intra-query parallelism, {graph_type} on {dataset}, max_degree={max_degree}', fontsize=14, y=1.2)
            if show_title:
                ax_speedup.set_title('Speedup over BFS', fontsize=label_font)
                ax_nodes.set_title('Normalized average #hops per search', fontsize=label_font)
                ax_recall.set_title('Recall R@10', fontsize=label_font)

        # save each subplot as a separate image
        for out_type in ['png', 'pdf']:
            plt.savefig('./images/dst_speedup_nodes_recall/dst_speedup_nodes_recall_{}_{}_MD{}_ef{}_{}.{}'.format(dataset, graph_type, max_degree, ef, suffix, out_type), transparent=False, dpi=200, bbox_inches="tight")

        # plt.show()

    def plot_each_subplot(mode='speedup'):
        # get three subplots, horizontally 
        fig, ax = plt.subplots(figsize=(5.5, 1.8))

        # matplotlib color map objects: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        # cmap = 'RdBu'
        # cmap = 'RdYlGn'
        # cmap = 'coolwarm'
        # cmap = 'bwr'
        # cmap = 'seismic'

        label_font = 13
        tick_font = 12
        legend_font = 12

        # Heatmap Document: https://seaborn.pydata.org/generated/seaborn.heatmap.html
        if mode == 'speedup':
            if suffix == "intra_query":
                cmap = 'YlGn'
            elif suffix == "inter_query":
                cmap = 'YlGnBu'
            seaborn.heatmap(speedup_array, cmap=cmap, annot=True, fmt=".2f", cbar_kws={})
            
        elif mode == 'nodes':
            seaborn.heatmap(normalized_node_array, cmap='OrRd', annot=True, fmt=".2f", cbar_kws={})
        elif mode == 'recall':
            seaborn.heatmap(recall_array, cmap='Purples', annot=True, fmt=".2f", cbar_kws={})
        elif mode == 'compute_util':
            if suffix == "intra_query":
                cmap = 'Greens'
            elif suffix == "inter_query":
                cmap = 'Blues'
            seaborn.heatmap(compute_util_array, cmap=cmap, annot=True, fmt=".2f", cbar_kws={})
        
        # ax_heatmap = seaborn.heatmap(data, cmap=cmap, cbar_kws={'label': 'colorbar title'})
        # set colorbar label size by hacking: https://stackoverflow.com/questions/48586738/seaborn-heatmap-colorbar-label-font-size
        ax.figure.axes[-1].yaxis.label.set_size(12)

        x_tick_labels = np.arange(1, max_mg + 1)
        y_tick_labels = np.arange(1, max_mc + 1)
        # 2**10 is on the very top in the heatmap, we want to put it down
        # y_tick_labels.reverse()

        ax.set_xticklabels(x_tick_labels)
        ax.set_yticklabels(y_tick_labels)
        ax.set_xlabel('max #groups in the pipeline (mg)', fontsize=label_font, labelpad=0)
        ax.set_ylabel('max #cand.\nper group (mc)', fontsize=label_font, labelpad=5)
        ax.tick_params(length=0, top=False, bottom=False, left=False, right=False, 
            labelleft=True, labelbottom=True)
        plt.yticks(rotation=0)
        # ax_speedup.ticklabel_format(axis='both', style='sci')
        # ax_speedup.set_yscale("log")


        # on the upper right corner, mark the dataset name
        # ax_recall.text(0.75, 1.12, f"{dataset}, {graph_type}", horizontalalignment='left', verticalalignment='center', transform=ax_recall.transAxes, fontsize=legend_font)

        # add plot title
        if mode == "speedup":
            if suffix == "inter_query" and show_title:
                ax.set_title('(b) Speedup over BFS (Across-query)', fontsize=label_font)
            elif suffix == "intra_query" and show_title:
                ax.set_title('(a) Speedup over BFS (Intra-query)', fontsize=label_font)
            for out_type in ['png', 'pdf']:
                plt.savefig('./images/dst_speedup_nodes_recall/dst_speedup_{}_{}_MD{}_ef{}_{}.{}'.format(dataset, graph_type, max_degree, ef, suffix, out_type), transparent=False, dpi=200, bbox_inches="tight")
        elif mode == "nodes":
            if show_title:
                ax.set_title('(c) Normalized avg #hops per search', fontsize=label_font)
            for out_type in ['png', 'pdf']:
                plt.savefig('./images/dst_speedup_nodes_recall/dst_nodes_{}_{}_MD{}_ef{}_{}.{}'.format(dataset, graph_type, max_degree, ef, suffix, out_type), transparent=False, dpi=200, bbox_inches="tight")
        elif mode == "recall":
            if show_title:
                ax.set_title('(f) Recall R@10', fontsize=label_font)
                ax.text(0.75, 1.12, f"{dataset}, {graph_type}", horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=label_font)
            for out_type in ['png', 'pdf']:
                plt.savefig('./images/dst_speedup_nodes_recall/dst_recall_{}_{}_MD{}_ef{}_{}.{}'.format(dataset, graph_type, max_degree, ef, suffix, out_type), transparent=False, dpi=200, bbox_inches="tight")
        if mode == "compute_util":
            if suffix == "inter_query" and show_title:
                ax.set_title('(e) Compute efficiency over BFS (Across-query)', fontsize=label_font)
            elif suffix == "intra_query" and show_title:
                ax.set_title('(d) Compute efficiency over BFS (Intra-query)', fontsize=label_font)
            for out_type in ['png', 'pdf']:
                plt.savefig('./images/dst_speedup_nodes_recall/dst_comp_util_{}_{}_MD{}_ef{}_{}.{}'.format(dataset, graph_type, max_degree, ef, suffix, out_type), transparent=False, dpi=200, bbox_inches="tight")

        # save each subplot as a separate image

    # plt.show() 

    # plot_all()
    plot_each_subplot('speedup')
    plot_each_subplot('nodes')
    plot_each_subplot('recall')
    plot_each_subplot('compute_util')

if __name__ == "__main__":
    # load dataframe
    df = pd.read_pickle(args.df_path)
    show_title = True
    if args.plot_all:
        # graph_types = ['HNSW']
        graph_types = ['HNSW', 'NSG']
        # datasets = ['SIFT1M']
        # datasets = ['SIFT1M', 'SIFT10M', 'SBERT1M']
        datasets = ['SIFT10M', 'Deep10M', 'SPACEV10M']
        max_degrees = [64]
        # max_degrees = [16, 32, 64]
        efs = [64]
        for graph_type in graph_types:
            for dataset in datasets:
                for max_degree in max_degrees:
                    for ef in efs:
                        plot_heatmap(df, graph_type, dataset, max_degree, ef, suffix=args.suffix, max_mg=args.max_mg, max_mc=args.max_mc, show_title=show_title)
    else:
        plot_heatmap(df, args.graph_type, args.dataset, args.max_degree, args.ef, suffix=args.suffix, max_mg=args.max_mg, max_mc=args.max_mc, show_title=True)
