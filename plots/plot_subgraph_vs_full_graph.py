import argparse
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns 

plt.rcParams['pdf.fonttype'] = 42
# Seaborn version >= 0.11.0

sns.set_theme(style="whitegrid")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="SIFT1M", help="the performance pickle file to save the dataframe")
parser.add_argument('--min_recall', type=float, default=0.95, help="the min recall range")
parser.add_argument('--max_recall', type=float, default=0.995, help="the max recall range")
parser.add_argument('--comparison_recall', type=float, default=0.99, help="compare the amount of workload at this recall")
args = parser.parse_args()

num_queries = 10000 # the pickle records total node counter of 10K queries

def plot_subgraph_total_workload(dataset='SIFT1M', max_degree=64, comparison_recall=0.99,
                 min_recall=0.95, max_recall=0.995, min_node_counter=0, max_node_counter=1e9):
        
    file_path = 'saved_perf_CPU/perf_subgraph_df.pickle'
    df = pd.read_pickle(file_path)

    # filter for sub mode
    sub_graph_num_list = [1, 2, 4, 8]#, 16]
    
    fig, ax_total = plt.subplots(1, 1, figsize=(3.4, 0.8), sharey=True)

    label_font = 11
    markersize = 6
    tick_font = 10
    legend_font = 10
    plots_total = []
    markers = ['o', '^', 'X', 's', 'D', 'P']

    recall_10_dict = dict()
    node_counter_dict = dict()
    
    for pid, sub_graph_num in enumerate(sub_graph_num_list):

        if sub_graph_num == 1:
            filtered_df = df[(df['dataset'] == dataset) &
                            (df['max_degree'] == max_degree) &
                            (df['mode'] == "full")]
        else:
            filtered_df = df[(df['dataset'] == dataset) &
                            (df['max_degree'] == max_degree) &
                            (df['mode'] == "sub") &
                            (df['sub_graph_num'] == sub_graph_num) ]
        
        # print(filtered_df)

        recall_10_list = []
        node_counter_list = []
        for idx, row in filtered_df.iterrows():
            recall_10 = row['recall_10']
            if recall_10 is None: 
                continue
            node_counter = row['node_counter'] / num_queries
            if recall_10 > min_recall and recall_10 < max_recall and node_counter > min_node_counter and node_counter < max_node_counter:
                recall_10_list.append(recall_10 * 100)
                node_counter_list.append(node_counter)
        
        recall_10_dict[sub_graph_num] = recall_10_list
        node_counter_dict[sub_graph_num] = node_counter_list
                
        if sub_graph_num == 1:
            label = "Full graph"
        else:
            label = f"{str(sub_graph_num)} sub-graphs" 

        # share y axis 
        plot_total = ax_total.plot(node_counter_list, np.array(recall_10_list), marker=markers[pid], markersize=markersize, label=label)
        plots_total.append(plot_total)

    for sub_graph_num in sub_graph_num_list:
        recall_10_list = recall_10_dict[sub_graph_num]
        node_counter_list = node_counter_dict[sub_graph_num]
        comparison_recall_idx = np.argmax(np.array(recall_10_list) > comparison_recall * 100)
        # print(comparison_recall_idx)
        # print(np.array(recall_10_list) > comparison_recall * 100)
        if sub_graph_num == 1:
            baseline_node_counter = node_counter_list[comparison_recall_idx]
            print("Baseline: Total workload: {:.2f}\t".format(baseline_node_counter))
        else:
            comparison_node_counter = node_counter_list[comparison_recall_idx]
            total_workload_ratio = comparison_node_counter / baseline_node_counter
            print("Subgraph num: {}\tTotal workload: {:.2f}\tTotal workload ratio: {:.2f} (theoretical speedup: {:.2f}))".format(
                sub_graph_num, comparison_node_counter, total_workload_ratio, sub_graph_num/total_workload_ratio))
            

    ax_total.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax_total.get_xaxis().set_visible(True)
    ax_total.set_xlabel('Total workload (visited nodes)', fontsize=label_font)
    ax_total.set_ylabel('R@10 (%)', fontsize=label_font)
    ax_total.legend(loc=(-0.1,1.05), fontsize=legend_font, ncol=2, frameon=False)
    ax_total.set_ylim(min_recall*100, max_recall*100)
    # ax_total.set_ylim(min_recall, max_recall)
    # ax_total.set_xlim(min_node_counter, max_node_counter)

    for out_type in ['png', 'pdf']:
        plt.savefig(f'./images/subgraph_vs_full_graph/subgraph_vs_full_graph_{dataset}.{out_type}', transparent=False, dpi=200, bbox_inches="tight")


def plot_subgraph_including_unbalanced_workload(dataset='SIFT1M', max_degree=64, comparison_recall=0.99,
                 min_recall=0.95, max_recall=0.995, min_node_counter=0, max_node_counter=1e9):
        
    file_path = 'saved_perf_CPU/perf_subgraph_df.pickle'
    df = pd.read_pickle(file_path)

    # filter for sub mode
    sub_graph_num_list = [1, 2, 4, 8]#, 16]
    
    fig, (ax_total, ax_per_partition) = plt.subplots(1, 2, figsize=(6.8, 1.2), sharey=True)

    label_font = 11
    markersize = 6
    tick_font = 10
    legend_font = 10
    plots_total = []
    plots_load_per_partition = []
    markers = ['o', '^', 'X', 's', 'D', 'P']

    recall_10_dict = dict()
    node_counter_dict = dict()
    average_max_node_counter_per_partition_dict = dict()

    for pid, sub_graph_num in enumerate(sub_graph_num_list):

        if sub_graph_num == 1:
            filtered_df = df[(df['dataset'] == dataset) &
                            (df['max_degree'] == max_degree) &
                            (df['mode'] == "full")]
        else:
            filtered_df = df[(df['dataset'] == dataset) &
                            (df['max_degree'] == max_degree) &
                            (df['mode'] == "sub") &
                            (df['sub_graph_num'] == sub_graph_num) ]
        
        # print(filtered_df)

        recall_10_list = []
        node_counter_list = []
        average_max_node_counter_per_partition_list = []
        for idx, row in filtered_df.iterrows():
            recall_10 = row['recall_10']
            if recall_10 is None: 
                continue
            node_counter = row['node_counter'] / num_queries
            node_counter_per_partition = row['node_counter_per_query'] # 2-d array (query_num, sub_graph_num) -> get the max of second dimension
            # print(node_counter_per_partition[:10])
            max_node_counter_per_partition = np.max(node_counter_per_partition, axis=1)
            # print(max_node_counter_per_partition[:10])
            average_max_node_counter_per_partition = np.average(max_node_counter_per_partition)
            if recall_10 > min_recall and recall_10 < max_recall and node_counter > min_node_counter and node_counter < max_node_counter:
                recall_10_list.append(recall_10 * 100)
                node_counter_list.append(node_counter)
                average_max_node_counter_per_partition_list.append(average_max_node_counter_per_partition)
        
        recall_10_dict[sub_graph_num] = recall_10_list
        node_counter_dict[sub_graph_num] = node_counter_list
        average_max_node_counter_per_partition_dict[sub_graph_num] = average_max_node_counter_per_partition_list
                
        if sub_graph_num == 1:
            label = "Full graph"
        else:
            label = f"{str(sub_graph_num)} subgraphs" 

        # share y axis 
        plot_total = ax_total.plot(node_counter_list, np.array(recall_10_list), marker=markers[pid], markersize=markersize, label=label)
        plot_load_per_partition = ax_per_partition.plot(average_max_node_counter_per_partition_list, np.array(recall_10_list), marker=markers[pid], markersize=markersize, label=label)
        plots_total.append(plot_total)
        plots_load_per_partition.append(plot_load_per_partition)

    for sub_graph_num in sub_graph_num_list:
        recall_10_list = recall_10_dict[sub_graph_num]
        node_counter_list = node_counter_dict[sub_graph_num]
        average_max_node_counter_per_partition_list = average_max_node_counter_per_partition_dict[sub_graph_num]
        comparison_recall_idx = np.argmax(np.array(recall_10_list) > comparison_recall * 100)
        # print(comparison_recall_idx)
        # print(np.array(recall_10_list) > comparison_recall * 100)
        if sub_graph_num == 1:
            baseline_node_counter = node_counter_list[comparison_recall_idx]
            baseline_max_node_counter_per_partition = average_max_node_counter_per_partition_list[comparison_recall_idx]
            print("Baseline: Total workload: {:.2f}\tPer partition workload: {:.2f}".format(
                baseline_node_counter, baseline_max_node_counter_per_partition
            ))
        else:
            comparison_node_counter = node_counter_list[comparison_recall_idx]
            comparison_max_node_counter_per_partition = average_max_node_counter_per_partition_list[comparison_recall_idx]
            total_workload_ratio = comparison_node_counter / baseline_node_counter
            per_partition_workload_ratio = comparison_max_node_counter_per_partition / baseline_max_node_counter_per_partition
            print("Subgraph num: {}\tTotal workload: {:.2f}\tPer partition workload: {:.2f}\tTotal workload ratio: {:.2f} (theoretical speedup: {:.2f})\tPer partition workload ratio: {:.2f} (real max speedup {:.2f})".format(
                sub_graph_num, comparison_node_counter, comparison_max_node_counter_per_partition, total_workload_ratio, sub_graph_num/total_workload_ratio, per_partition_workload_ratio, 1/per_partition_workload_ratio
            ))
            

    ax_total.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax_total.get_xaxis().set_visible(True)
    ax_total.set_xlabel('(a) Total workload (visited nodes)', fontsize=label_font)
    ax_total.set_ylabel('R@10 (%)', fontsize=label_font)
    ax_total.legend(loc=(-0.2,1.05), fontsize=legend_font, ncol=4, frameon=False)
    # ax_total.set_ylim(min_recall, max_recall)
    # ax_total.set_xlim(min_node_counter, max_node_counter)

    ax_per_partition.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    # ax_per_partition.set_ylabel('R@10 (%)', fontsize=label_font)
    ax_per_partition.set_xlabel('(b) Max workload per subgraph', fontsize=label_font)

    # lower right corner of ax_per_partition and ax_total, print dataaset name and graph name (HNSW)
    # ax_total.text(0.99, 0.01, f'HNSW on {dataset}', verticalalignment='bottom', horizontalalignment='right', 
    #                       transform =ax_total.transAxes, fontsize=tick_font)
    ax_per_partition.text(0.99, 0.01, f'HNSW on {dataset}', verticalalignment='bottom', horizontalalignment='right', 
                          transform =ax_per_partition.transAxes, fontsize=tick_font)
    for out_type in ['png', 'pdf']:
        plt.savefig(f'./images/subgraph_vs_full_graph/subgraph_vs_full_graph_{dataset}.{out_type}', transparent=False, dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    plot_subgraph_total_workload(dataset=args.dataset, min_recall=args.min_recall, max_recall=args.max_recall, comparison_recall=args.comparison_recall)
    # plot_subgraph_including_unbalanced_workload()
