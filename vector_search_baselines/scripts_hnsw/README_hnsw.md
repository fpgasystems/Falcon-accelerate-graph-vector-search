# HNSW scripts

## Construct HNSW Indexes

First, construct HNSW:

```
# SIFT1M
python construct_and_search_hnsw.py --dbname SIFT1M --ef_construction 128 --MD 64 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SIFT1M --ef_construction 128 --MD 32 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SIFT1M --ef_construction 128 --MD 16 --hnsw_path ../data/CPU_hnsw_indexes

# SIFT10M
python construct_and_search_hnsw.py --dbname SIFT10M --ef_construction 128 --MD 64 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SIFT10M --ef_construction 128 --MD 32 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SIFT10M --ef_construction 128 --MD 16 --hnsw_path ../data/CPU_hnsw_indexes

# Deep1M
python construct_and_search_hnsw.py --dbname Deep1M --ef_construction 128 --MD 64 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname Deep1M --ef_construction 128 --MD 32 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname Deep1M --ef_construction 128 --MD 16 --hnsw_path ../data/CPU_hnsw_indexes

# Deep10M
python construct_and_search_hnsw.py --dbname Deep10M --ef_construction 128 --MD 64 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname Deep10M --ef_construction 128 --MD 32 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname Deep10M --ef_construction 128 --MD 16 --hnsw_path ../data/CPU_hnsw_indexes

# SBERT1M
python construct_and_search_hnsw.py --dbname SBERT1M --ef_construction 128 --MD 64 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SBERT1M --ef_construction 128 --MD 32 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SBERT1M --ef_construction 128 --MD 16 --hnsw_path ../data/CPU_hnsw_indexes

# SPACEV1M
python construct_and_search_hnsw.py --dbname SPACEV1M --ef_construction 128 --MD 64 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SPACEV1M --ef_construction 128 --MD 32 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SPACEV1M --ef_construction 128 --MD 16 --hnsw_path ../data/CPU_hnsw_indexes

# SPACEV10M
python construct_and_search_hnsw.py --dbname SPACEV10M --ef_construction 128 --MD 64 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SPACEV10M --ef_construction 128 --MD 32 --hnsw_path ../data/CPU_hnsw_indexes
python construct_and_search_hnsw.py --dbname SPACEV10M --ef_construction 128 --MD 16 --hnsw_path ../data/CPU_hnsw_indexes
```

Second: convert to FPGA format:

```
# SIFT1M
python hnsw_to_FPGA.py --dbname SIFT1M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT1M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT1M_MD64
python hnsw_to_FPGA.py --dbname SIFT1M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT1M_index_MD32.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT1M_MD32
python hnsw_to_FPGA.py --dbname SIFT1M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT1M_index_MD16.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT1M_MD16

# SIFT10M
python hnsw_to_FPGA.py --dbname SIFT10M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT10M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT10M_MD64
python hnsw_to_FPGA.py --dbname SIFT10M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT10M_index_MD32.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT10M_MD32
python hnsw_to_FPGA.py --dbname SIFT10M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT10M_index_MD16.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT10M_MD16

# Deep1M
python hnsw_to_FPGA.py --dbname Deep1M --CPU_index_path ../data/CPU_hnsw_indexes/Deep1M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/Deep1M_MD64
python hnsw_to_FPGA.py --dbname Deep1M --CPU_index_path ../data/CPU_hnsw_indexes/Deep1M_index_MD32.bin --FPGA_index_path ../data/FPGA_hnsw/Deep1M_MD32
python hnsw_to_FPGA.py --dbname Deep1M --CPU_index_path ../data/CPU_hnsw_indexes/Deep1M_index_MD16.bin --FPGA_index_path ../data/FPGA_hnsw/Deep1M_MD16

# Deep10M
python hnsw_to_FPGA.py --dbname Deep10M --CPU_index_path ../data/CPU_hnsw_indexes/Deep10M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/Deep10M_MD64
python hnsw_to_FPGA.py --dbname Deep10M --CPU_index_path ../data/CPU_hnsw_indexes/Deep10M_index_MD32.bin --FPGA_index_path ../data/FPGA_hnsw/Deep10M_MD32
python hnsw_to_FPGA.py --dbname Deep10M --CPU_index_path ../data/CPU_hnsw_indexes/Deep10M_index_MD16.bin --FPGA_index_path ../data/FPGA_hnsw/Deep10M_MD16

# SBERT1M
python hnsw_to_FPGA.py --dbname SBERT1M --CPU_index_path ../data/CPU_hnsw_indexes/SBERT1M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/SBERT1M_MD64
python hnsw_to_FPGA.py --dbname SBERT1M --CPU_index_path ../data/CPU_hnsw_indexes/SBERT1M_index_MD32.bin --FPGA_index_path ../data/FPGA_hnsw/SBERT1M_MD32
python hnsw_to_FPGA.py --dbname SBERT1M --CPU_index_path ../data/CPU_hnsw_indexes/SBERT1M_index_MD16.bin --FPGA_index_path ../data/FPGA_hnsw/SBERT1M_MD16

# SPACEV1M
python hnsw_to_FPGA.py --dbname SPACEV1M --CPU_index_path ../data/CPU_hnsw_indexes/SPACEV1M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/SPACEV1M_MD64
python hnsw_to_FPGA.py --dbname SPACEV1M --CPU_index_path ../data/CPU_hnsw_indexes/SPACEV1M_index_MD32.bin --FPGA_index_path ../data/FPGA_hnsw/SPACEV1M_MD32
python hnsw_to_FPGA.py --dbname SPACEV1M --CPU_index_path ../data/CPU_hnsw_indexes/SPACEV1M_index_MD16.bin --FPGA_index_path ../data/FPGA_hnsw/SPACEV1M_MD16

# SPACEV10M
python hnsw_to_FPGA.py --dbname SPACEV10M --CPU_index_path ../data/CPU_hnsw_indexes/SPACEV10M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/SPACEV10M_MD64
python hnsw_to_FPGA.py --dbname SPACEV10M --CPU_index_path ../data/CPU_hnsw_indexes/SPACEV10M_index_MD32.bin --FPGA_index_path ../data/FPGA_hnsw/SPACEV10M_MD32
python hnsw_to_FPGA.py --dbname SPACEV10M --CPU_index_path ../data/CPU_hnsw_indexes/SPACEV10M_index_MD16.bin --FPGA_index_path ../data/FPGA_hnsw/SPACEV10M_MD16
```

## Evaluate CPU latency & throughput

Before running the following commands, make sure the configurations in the scripts are properly set:, e.g., 
```
    max_degree_list = [64]
    ef_list = [64]
    omp_list = [1] # 1 = enable; 0 = disable
    batch_size_list = [1, 2, 4, 8, 16, 10000]
```

Now run (make sure `max_cores` is properly set):
```
python run_all_hnsw_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_search_bin_path ../hnswlib/build/main --perf_df_path perf_df_hnsw_cpu.pickle --dataset SIFT1M --max_cores 16  --nruns 3
python run_all_hnsw_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_search_bin_path ../hnswlib/build/main --perf_df_path perf_df_hnsw_cpu.pickle --dataset SIFT10M --max_cores 16  --nruns 3
python run_all_hnsw_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_search_bin_path ../hnswlib/build/main --perf_df_path perf_df_hnsw_cpu.pickle --dataset Deep1M --max_cores 16  --nruns 3
python run_all_hnsw_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_search_bin_path ../hnswlib/build/main --perf_df_path perf_df_hnsw_cpu.pickle --dataset Deep10M --max_cores 16  --nruns 3
python run_all_hnsw_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_search_bin_path ../hnswlib/build/main --perf_df_path perf_df_hnsw_cpu.pickle --dataset SPACEV1M --max_cores 16  --nruns 3
python run_all_hnsw_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_search_bin_path ../hnswlib/build/main --perf_df_path perf_df_hnsw_cpu.pickle --dataset SPACEV10M --max_cores 16  --nruns 3
cp perf_df_hnsw_cpu.pickle /mnt/scratch/wenqi/graph-vector-search-on-FPGA/plots/saved_perf_CPU/hnsw.pickle
```

## Evaluate CPU HNSW energy 


Use two terminals, one for executing the program, the other for tracking energy.

```
# SIFT10M
python run_all_hnsw_inf_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_inf_search_bin_path ../hnswlib/build/inf_search \
     --dataset SIFT10M --max_cores 16 --max_degree 64 --ef 64 --omp 1 --batch_size 1
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_hnsw_SIFT10M_batch_1
python run_all_hnsw_inf_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_inf_search_bin_path ../hnswlib/build/inf_search \
     --dataset SIFT10M --max_cores 16 --max_degree 64 --ef 64 --omp 1 --batch_size 16
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_hnsw_SIFT10M_batch_16
python run_all_hnsw_inf_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_inf_search_bin_path ../hnswlib/build/inf_search \
     --dataset SIFT10M --max_cores 16 --max_degree 64 --ef 64 --omp 1 --batch_size 10000
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_hnsw_SIFT10M_batch_10000

# Deep10M
python run_all_hnsw_inf_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_inf_search_bin_path ../hnswlib/build/inf_search \
     --dataset Deep10M --max_cores 16 --max_degree 64 --ef 64 --omp 1 --batch_size 1
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_hnsw_Deep10M_batch_1
python run_all_hnsw_inf_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_inf_search_bin_path ../hnswlib/build/inf_search \
     --dataset Deep10M --max_cores 16 --max_degree 64 --ef 64 --omp 1 --batch_size 16
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_hnsw_Deep10M_batch_16
python run_all_hnsw_inf_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_inf_search_bin_path ../hnswlib/build/inf_search \
     --dataset Deep10M --max_cores 16 --max_degree 64 --ef 64 --omp 1 --batch_size 10000
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_hnsw_Deep10M_batch_10000

# SPACEV10M
python run_all_hnsw_inf_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_inf_search_bin_path ../hnswlib/build/inf_search \
     --dataset SPACEV10M --max_cores 16 --max_degree 64 --ef 64 --omp 1 --batch_size 1
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_hnsw_SPACEV10M_batch_1
python run_all_hnsw_inf_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_inf_search_bin_path ../hnswlib/build/inf_search \
     --dataset SPACEV10M --max_cores 16 --max_degree 64 --ef 64 --omp 1 --batch_size 16
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_hnsw_SPACEV10M_batch_16
python run_all_hnsw_inf_search.py --hnsw_index_path ../data/CPU_hnsw_indexes --hnsw_inf_search_bin_path ../hnswlib/build/inf_search \
     --dataset SPACEV10M --max_cores 16 --max_degree 64 --ef 64 --omp 1 --batch_size 10000
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_hnsw_SPACEV10M_batch_10000
```

## Explore the relationship between a big graph versus several sub-graphs

```
python subgraph_vs_full_graph_hnsw.py --dbname SIFT1M --ef_construction 128 --MD 64 \
	--hnsw_path ../data/CPU_hnsw_indexes --subgraph_result_path ../data/sub_graph_results

python subgraph_vs_full_graph_hnsw.py --dbname Deep1M --ef_construction 128 --MD 64 \
	--hnsw_path ../data/CPU_hnsw_indexes --subgraph_result_path ../data/sub_graph_results

python subgraph_vs_full_graph_hnsw.py --dbname SPACEV1M --ef_construction 128 --MD 64 \
	--hnsw_path ../data/CPU_hnsw_indexes --subgraph_result_path ../data/sub_graph_results
```
