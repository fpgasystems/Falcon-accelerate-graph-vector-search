import numpy as np
import os 
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter

plt.style.use('seaborn-pastel')
# plt.style.use('seaborn-colorblind')

plt.rcParams['pdf.fonttype'] = 42

default_colors = []
for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
    default_colors.append(color["color"])
    print(color["color"], type(color["color"]))
print(default_colors[0], type(default_colors[0]))


def plot_distance_over_steps_subplots(dataset="SIFT1M", graph_type="HNSW", mc_mg_groups = [(1, 1), (4, 1), (2, 2)]):

    folder_name = 'saved_distances_over_steps'

    n_subplots = len(mc_mg_groups) 
    fig, ax = plt.subplots(3, 1, figsize=(6, 3), sharex=True,) # share x axis

    # distances between plots
    plt.subplots_adjust(hspace=0.25)

    # /// debug_out_per_query_dists_ids format
    # /// -1 (as start) -> dtype = float / int, depends on dist / id
    # // dist, id of 1st step
    # // dist, id of 2nd step
    # // ...
    # // dist, id of last step
    # /// -2 (as finish)

    label_font = 11
    markersize = 0.1
    tick_font = 8
    title_font = 11

    # set x lim
    x_max = None
    y_max = None
    
    algo_names = []
    for pid, (mc, mg) in enumerate(mc_mg_groups):
        fname_dist = os.path.join(folder_name, f'per_query_dists_{dataset}_{graph_type}_mc{mc}_mg{mg}.float')
        fname_cand_dist = os.path.join(folder_name, f'per_query_cand_dists_{dataset}_{graph_type}_mc{mc}_mg{mg}.float')
        fname_cand_num_neighbors = os.path.join(folder_name, f'per_query_cand_num_neighbors_{dataset}_{graph_type}_mc{mc}_mg{mg}.int')
        data_raw_dist = np.fromfile(fname_dist, dtype=np.float32)
        data_raw_cand_dist = np.fromfile(fname_cand_dist, dtype=np.float32)
        data_raw_cand_num_neighbors = np.fromfile(fname_cand_num_neighbors, dtype=np.int32)
        dist_over_steps = []
        cand_dist_over_steps = []
        cand_num_neighbors_over_steps = []
        cand_dist_step_ids = [0]
        qid = 1 # query id
        # qid = 10 # query id
        q_cnt = 0
        for j, dist in enumerate(list(data_raw_dist)):
            if dist == -1:
                continue
            else:
                if dist == -2:
                    if q_cnt == qid:
                        break
                    else:
                        q_cnt += 1
                elif q_cnt == qid:
                    dist_over_steps.append(dist)

        q_cnt = 0
        for j, cand_dist in enumerate(list(data_raw_cand_dist)):
            if cand_dist == -1:
                continue
            else:
                if cand_dist == -2:
                    if q_cnt == qid:
                        break
                    else:
                        q_cnt += 1
                elif q_cnt == qid:
                    cand_dist_over_steps.append(cand_dist)
        
        q_cnt = 0
        for j, cand_num_neighbors in enumerate(list(data_raw_cand_num_neighbors)):
            if cand_num_neighbors == -1:
                continue
            else:
                if cand_num_neighbors == -2:
                    if q_cnt == qid:
                        break
                    else:
                        q_cnt += 1
                elif q_cnt == qid:
                    cand_num_neighbors_over_steps.append(cand_num_neighbors)
                    cand_dist_step_ids.append(cand_dist_step_ids[-1] + cand_num_neighbors)
        cand_dist_step_ids = cand_dist_step_ids[:-1]
        
        assert len(cand_dist_over_steps) == len(cand_num_neighbors_over_steps) and len(cand_dist_over_steps) == len(cand_dist_step_ids)
        # print("Sum over neighbors of candidates: ", np.sum(cand_num_neighbors_over_steps))
        # print("Real evaluated number of nodes: ", len(dist_over_steps))
        # print(cand_num_neighbors_over_steps)
        assert np.sum(cand_num_neighbors_over_steps) == len(dist_over_steps)
        # assert np.sum(cand_num_neighbors_over_steps) <= 1.05 * len(dist_over_steps) and np.sum(cand_num_neighbors_over_steps) >= 0.95 * len(dist_over_steps)

        # set title
        if mc == 1 and mg == 1:
            algo = "Best-First Search"
            # algo = "(a) Best-First Search"
            color = default_colors[0]
        elif mc > 1 and mg == 1:
            # algo = "(b) Multi-Candidate Search" + f" (mc={mc})"
            algo = "Multi-Candidate Search" + f" (mc={mc})"
            color = default_colors[2]
        elif mg > 1:
            # algo = "(c) Delayed-Synchronization Traversal" + f" (mc={mc}, mg={mg})"
            algo = "Delayed-Synchronization Traversal" + f" (mc={mc}, mg={mg})"
            color = default_colors[1]
        algo_names.append(algo)

        x = np.arange(len(dist_over_steps)) + 1
        print(len(x))
        plot_neighbors = ax[pid].scatter(x, dist_over_steps, marker='o', s=markersize, color=color)
        # reduce marker sizes
        plot_cand = ax[pid].scatter(cand_dist_step_ids, cand_dist_over_steps, marker='x', s=15, color='#BBBBBB')

        # find the 10 nearest neighbor and their indexes
        min_dist = []
        min_dist_idx = []
        dist_over_steps_with_idx = list(enumerate(dist_over_steps))
        dist_over_steps_with_idx.sort(key=lambda x: x[1])
        for i in range(10):
            min_dist.append(dist_over_steps_with_idx[i][1])
            min_dist_idx.append(dist_over_steps_with_idx[i][0])
        plot_nearest_neighbors = ax[pid].scatter(min_dist_idx, min_dist, marker='*', s=20, color='#555555')

        # plot1 = ax.plot(data_x, data_y_plot1, marker='^', markersize=markersize) # color=color_plot1,
        # ax.legend([plot0[0], plot1[0]], ["plot0_legend", "plot1_legend"], loc='upper right', fontsize=label_font)
        # ax[pid].get_xaxis().set_visible(True)
        if pid == n_subplots - 1:
            # ax[pid].tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
            ax[pid].set_xlabel('Traversal procedure (number of visited nodes)', fontsize=label_font)
        ax[pid].set_ylabel('Distance\nto query', fontsize=label_font)
        # set y as exponential
        ax[pid].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # set x lim
        if x_max is None and y_max is None:
            x_max = 1.02 * len(dist_over_steps)
            y_max = 1.2 * np.max(dist_over_steps)
        x_max = max(1.02 * len(dist_over_steps), x_max)
        for tmp_pid in range(pid + 1):
            ax[tmp_pid].set_xlim(0, x_max)
            ax[tmp_pid].set_ylim(0, y_max)

        # set y tick font
        # ax[pid].tick_params(axis='y', labelsize=tick_font)

        # find nearest neighbor in dist_over_steps, and add an arrow pointing to it
        # ax[pid].annotate(f'1st-NN', xy=(min_dist_idx + 1, min_dist), xytext=(min_dist_idx + 1, y_max * 0.7),
        #             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=label_font, horizontalalignment='center', verticalalignment='top', )

        # if pid == 0:
        #     ax[pid].set_title(f"Traversal processs of the same query (Dataset: {dataset}, Graph: {graph_type})", fontsize=title_font)

    for pid, (mc, mg) in enumerate(mc_mg_groups):
        ax[pid].text(0.98 * x_max, 0.95 * y_max, algo_names[pid], fontsize=label_font, verticalalignment='top', horizontalalignment='right')

        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        # plt.rcParams.update({'figure.autolayout': True})

    for out_type in ['png', 'pdf']:
        plt.savefig(f'./images/distance_over_steps/distances_over_steps_all_{dataset}_{graph_type}.{out_type}', transparent=False, dpi=200, bbox_inches="tight")
    # plt.show()

def plot_distance_over_steps(dataset="SIFT1M", graph_type="HNSW", mc=1, mg=1):
    
    folder_name = 'saved_distances_over_steps'
    fname_dist = os.path.join(folder_name, f'per_query_dists_{dataset}_{graph_type}_mc{mc}_mg{mg}.float')
    data_raw_dist = np.fromfile(fname_dist, dtype=np.float32)
    dist_over_steps = []

    # /// debug_out_per_query_dists_ids format
    # /// -1 (as start) -> dtype = float / int, depends on dist / id
    # // dist, id of 1st step
    # // dist, id of 2nd step
    # // ...
    # // dist, id of last step
    # /// -2 (as finish)

    qid = 1 # query id
    q_cnt = 0
    for i, dist in enumerate(list(data_raw_dist)):
        if dist == -1:
            continue
        else:
            if dist == -2:
                if q_cnt == qid:
                    break
                else:
                    q_cnt += 1
            elif q_cnt == qid:
                dist_over_steps.append(dist)

    fig, ax = plt.subplots(1, 1, figsize=(6, 1.2))

    label_font = 10
    markersize = 1
    tick_font = 10
    title_font = 11


    # set title
    if mc == 1 and mg == 1:
        algo = "Best-First Search"
        color = default_colors[0]
    elif mc > 1 and mg == 1:
        algo = "Multi-Candidate Search" + f" (mc={mc})"
        color = default_colors[1]
    elif mg > 1:
        algo = "Delayed-Synchronization Traversal" + f" (mc={mc}, mg={mg})"
        color = default_colors[2]
    
    print("mc: {}, mg: {}, total steps: {}".format(mc, mg, len(dist_over_steps)))

    x = np.arange(len(dist_over_steps)) + 1
    plot0 = ax.scatter(x, dist_over_steps, marker='o', s=markersize, color=color)
    # reduce marker sizes


    # plot1 = ax.plot(data_x, data_y_plot1, marker='^', markersize=markersize) # color=color_plot1,
    # ax.legend([plot0[0], plot1[0]], ["plot0_legend", "plot1_legend"], loc='upper right', fontsize=label_font)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel('Steps (each is a visited node)', fontsize=label_font)
    ax.set_ylabel('Distance to query', fontsize=label_font)
    # set y as exponential
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # set x lim
    x_max = 2050
    y_max = 4e+5
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # find nearest neighbor in dist_over_steps, and add an arrow pointing to it
    # find the nearest neighbor
    min_dist = np.min(dist_over_steps)
    min_dist_idx = np.argmin(dist_over_steps)
    ax.annotate(f'Nearest Neighbor', xy=(min_dist_idx + 1, min_dist), xytext=(min_dist_idx + 1, y_max * 0.75),
                arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=label_font, horizontalalignment='center', verticalalignment='top', )

    ax.set_title(algo, fontsize=title_font)

    ax.text(0.95 * x_max, 0.95 * y_max, f"Dataset: {dataset}, Graph: {graph_type}", fontsize=label_font, verticalalignment='top', horizontalalignment='right')

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # plt.rcParams.update({'figure.autolayout': True})

    for out_type in ['png', 'pdf']:
        plt.savefig(f'./images/distance_over_steps/distances_over_steps_{dataset}_{graph_type}_mc{mc}_mg{mg}.{out_type}', transparent=False, dpi=200, bbox_inches="tight")
    # plt.show()

if __name__ == "__main__":
    mc_mg_groups = [(1, 1), (4, 1), (2, 2)]
    
    datasets = ["SIFT1M", "Deep1M", "SPACEV1M"]
    graph_types = ["HNSW", "NSG"]
    for dataset in datasets:
        for graph_type in graph_types:
            plot_distance_over_steps_subplots(dataset=dataset, graph_type=graph_type, mc_mg_groups=mc_mg_groups)
    
    # for mc, mg in mc_mg_groups:
    #     plot_distance_over_steps(mc=mc, mg=mg)
