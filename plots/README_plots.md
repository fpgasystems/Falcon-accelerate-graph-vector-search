# Plots

This folder contains all the plotting scripts as well as scripts used for printing performance information.

## Files and Folders

`saved_...` are the data saved for plotting.

`calculate_silicon_efficiency.py` is the script for printing silicon efficiency of different traversals. 

`plot_dst_speedup_across_settings.py` plots the speed of of DST over BFS, given different graphs, degrees, and datasets.

`plot_dst_speedup_nodes_recall.py` plots the heatmap of different DST configurations, reporting performance, hops per query, and recall.

`plot_intra_query_chan_scalability.py` shows the performance scalability of intra-query parallelism, given different numbers of processing pipelines.

`plot_latency_CPU_GPU_FPGA.py` shows the latency comparison between CPUs, GPUs, and FPGAs.

`plot_throughput_CPU_GPU_FPGA.py` shows the throughput comparison between CPUs, GPUs, and FPGAs.

(unsed) `plot_intra_vs_inter_performance.py` shows the performance cross-over points between intra and Across-query parallelism, without network latency. 

(unused) `plot_latency_FPGA.py` shows the latency distribution of only FPGAs.

## Commands for plotting all the figures for the paper

Outputs are stored in `images`.

DST convergence: 
```
python plot_distance_over_steps_different_traversals.py 
```

DST speedup over BFS:
```
python plot_dst_speedup_across_settings.py \
	--df_path_inter_query ../perf_test_scripts/saved_df/throughput_FPGA_inter_query_4_chan.pickle \
	--df_path_intra_query ../perf_test_scripts/saved_df/throughput_FPGA_intra_query_4_chan.pickle
```

DST heatmap (performance, #hops, recall):
```
# plot all (hard-coded) settings: intra-query
python plot_dst_speedup_nodes_recall.py --plot_all 1 --df_path ../perf_test_scripts/saved_df/throughput_FPGA_intra_query_4_chan.pickle --suffix intra_query --max_mc 4 --max_mg 7
# plot all (hard-coded) settings: Across-query
python plot_dst_speedup_nodes_recall.py --plot_all 1 --df_path ../perf_test_scripts/saved_df/throughput_FPGA_inter_query_4_chan.pickle --suffix inter_query --max_mc 4 --max_mg 7
```

Intra-query parallel scalability:
```
python plot_intra_query_chan_scalability.py \
	--df_path_1_chan ../perf_test_scripts/saved_df/latency_FPGA_intra_query_1_chan.pickle \
	--df_path_2_chan ../perf_test_scripts/saved_df/latency_FPGA_intra_query_2_chan.pickle \
	--df_path_4_chan ../perf_test_scripts/saved_df/latency_FPGA_intra_query_4_chan.pickle
```

Latency CPU, GPU, FPGA:
```
python plot_latency_CPU_GPU_FPGA.py 
```

Throughput CPU, GPU, FPGA:
```
python plot_throughput_CPU_GPU_FPGA.py 
```

Energy consumption:
```
python print_energy.py 
```

Subgraph vs full graph search:

```
python plot_subgraph_vs_full_graph.py --dataset SPACEV1M --min_recall 0.85 --max_recall 0.95 --comparison_recall 0.9
python plot_subgraph_vs_full_graph.py --dataset SIFT1M --min_recall 0.96 --max_recall 0.995 --comparison_recall 0.99
python plot_subgraph_vs_full_graph.py --dataset Deep1M --min_recall 0.95 --max_recall 0.995 --comparison_recall 0.9
```

DLRM e2e:
```
python plot_dlrm_profile.py
```

RAG e2e:
```
python plot_llm_profile.py
```
