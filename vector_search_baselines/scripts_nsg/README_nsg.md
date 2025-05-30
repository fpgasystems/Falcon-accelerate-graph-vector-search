# NSG scripts

## We applied some changes to the NSG folder

1. To avoid recall randomness, we removed the additional random initial candidates at the start of NSG search. Currently it's safe to assert that such modification has little impact on recall.

2. To ensure that the generated Navigating Spread-out Graph maintains a maximum degree no more than $R$, we introduced a new condition for each edge addition $e(u,v)$ that $d(u)$ should be no more than $R$. Otherwise, the program proceeds to traverse the candidates or randomly pick a node whose degree does not exceed R.

3. We support multiple dataset formats (see `include/load_data.h`) and have adapted them in the NSG index construction and search.

## Construct NSG Indexes

First: construct the exact kNN graph using GPU faiss:

```
python construct_faiss_knn.py --dbname SIFT1M --construct_K 200 --output_path ../data/CPU_knn_graphs
python construct_faiss_knn.py --dbname SIFT10M --construct_K 200 --output_path ../data/CPU_knn_graphs
python construct_faiss_knn.py --dbname Deep1M --construct_K 200 --output_path ../data/CPU_knn_graphs
python construct_faiss_knn.py --dbname Deep10M --construct_K 200 --output_path ../data/CPU_knn_graphs
python construct_faiss_knn.py --dbname SBERT1M --construct_K 200 --output_path ../data/CPU_knn_graphs
```

Second: construct NSG

```
python run_all_nsg_construct_and_search.py --mode construct --input_knng_path ../data/CPU_knn_graphs --output_nsg_path ../data/CPU_NSG_index --nsg_con_bin_path ../nsg/build/tests/test_nsg_index --dataset SIFT1M

python run_all_nsg_construct_and_search.py --mode construct --input_knng_path ../data/CPU_knn_graphs --output_nsg_path ../data/CPU_NSG_index --nsg_con_bin_path ../nsg/build/tests/test_nsg_index --dataset SIFT10M

python run_all_nsg_construct_and_search.py --mode construct --input_knng_path ../data/CPU_knn_graphs --output_nsg_path ../data/CPU_NSG_index --nsg_con_bin_path ../nsg/build/tests/test_nsg_index --dataset Deep1M

python run_all_nsg_construct_and_search.py --mode construct --input_knng_path ../data/CPU_knn_graphs --output_nsg_path ../data/CPU_NSG_index --nsg_con_bin_path ../nsg/build/tests/test_nsg_index --dataset Deep10M

python run_all_nsg_construct_and_search.py --mode construct --input_knng_path ../data/CPU_knn_graphs --output_nsg_path ../data/CPU_NSG_index --nsg_con_bin_path ../nsg/build/tests/test_nsg_index --dataset SPACEV1M

python run_all_nsg_construct_and_search.py --mode construct --input_knng_path ../data/CPU_knn_graphs --output_nsg_path ../data/CPU_NSG_index --nsg_con_bin_path ../nsg/build/tests/test_nsg_index --dataset SPACEV10M

# python run_all_nsg_construct_and_search.py --mode construct --input_knng_path ../data/CPU_knn_graphs --output_nsg_path ../data/CPU_NSG_index --nsg_con_bin_path ../nsg/build/tests/test_nsg_index --dataset SBERT1M
```


Third: convert to FPGA format

```
# SIFT1M
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SIFT1M --max_degree 64
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SIFT1M --max_degree 32
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SIFT1M --max_degree 16

# SIFT10M
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SIFT10M --max_degree 64
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SIFT10M --max_degree 32
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SIFT10M --max_degree 16

# Deep1M
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname Deep1M --max_degree 64
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname Deep1M --max_degree 32
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname Deep1M --max_degree 16

# Deep10M
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname Deep10M --max_degree 64
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname Deep10M --max_degree 32
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname Deep10M --max_degree 16

# SPACEV1M
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SPACEV1M --max_degree 64
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SPACEV1M --max_degree 32
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SPACEV1M --max_degree 16

# SPACEV10M
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SPACEV10M --max_degree 64
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SPACEV10M --max_degree 32
python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SPACEV10M --max_degree 16

# SBERT1M
# python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SBERT1M --max_degree 64
# python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SBERT1M --max_degree 32
# python nsg_to_FPGA.py --nsg_path ../data/CPU_NSG_index --FPGA_index_path /mnt/scratch/wenqi/hnsw_experiments/data/FPGA_NSG --dbname SBERT1M --max_degree 16
```

## Evaluate CPU latency & throughput

Before running the following commands, make sure the configurations in the scripts are properly set:, e.g., 
```
	search_L_list = [64] # the recall result shows that this is actually equivalent to ef=64 in HNSW
	omp_list = [1] # 1 = enable; 0 = disable
	batch_size_list = [1, 2, 4, 8, 16, 10000]
	construct_R_list = [64] # R means max degree
```

Now run (make sure `max_cores` is properly set):
```
python run_all_nsg_construct_and_search.py --mode search --nsg_search_bin_path ../nsg/build/tests/test_nsg_optimized_search --dataset SIFT1M --perf_df_path perf_df_nsg_cpu.pickle --max_cores 16 --nruns 3
python run_all_nsg_construct_and_search.py --mode search --nsg_search_bin_path ../nsg/build/tests/test_nsg_optimized_search --dataset SIFT10M --perf_df_path perf_df_nsg_cpu.pickle --max_cores 16 --nruns 3
python run_all_nsg_construct_and_search.py --mode search --nsg_search_bin_path ../nsg/build/tests/test_nsg_optimized_search --dataset Deep1M --perf_df_path perf_df_nsg_cpu.pickle --max_cores 16 --nruns 3
python run_all_nsg_construct_and_search.py --mode search --nsg_search_bin_path ../nsg/build/tests/test_nsg_optimized_search --dataset Deep10M --perf_df_path perf_df_nsg_cpu.pickle --max_cores 16 --nruns 3
python run_all_nsg_construct_and_search.py --mode search --nsg_search_bin_path ../nsg/build/tests/test_nsg_optimized_search --dataset SPACEV1M --perf_df_path perf_df_nsg_cpu.pickle --max_cores 16 --nruns 3
python run_all_nsg_construct_and_search.py --mode search --nsg_search_bin_path ../nsg/build/tests/test_nsg_optimized_search --dataset SPACEV10M --perf_df_path perf_df_nsg_cpu.pickle --max_cores 16 --nruns 3
cp perf_df_nsg_cpu.pickle /mnt/scratch/wenqi/graph-vector-search-on-FPGA/plots/saved_perf_CPU/nsg.pickle
```

## Evaluate CPU NSG energy 

Use two terminals, one for executing the program, the other for tracking energy.

```
# SIFT10M
python run_all_nsg_inf_search.py --nsg_inf_search_bin_path ../nsg/build/tests/test_nsg_optimized_inf_search --dataset SIFT10M --max_cores 16 --construct_R 64 --search_L 64 --omp 1 --batch_size 1
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_nsg_SIFT10M_batch_1
python run_all_nsg_inf_search.py --nsg_inf_search_bin_path ../nsg/build/tests/test_nsg_optimized_inf_search --dataset SIFT10M --max_cores 16 --construct_R 64 --search_L 64 --omp 1 --batch_size 16
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_nsg_SIFT10M_batch_16
python run_all_nsg_inf_search.py --nsg_inf_search_bin_path ../nsg/build/tests/test_nsg_optimized_inf_search --dataset SIFT10M --max_cores 16 --construct_R 64 --search_L 64 --omp 1 --batch_size 10000
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_nsg_SIFT10M_batch_10000

# Deep10M
python run_all_nsg_inf_search.py --nsg_inf_search_bin_path ../nsg/build/tests/test_nsg_optimized_inf_search --dataset Deep10M --max_cores 16 --construct_R 64 --search_L 64 --omp 1 --batch_size 1
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_nsg_Deep10M_batch_1
python run_all_nsg_inf_search.py --nsg_inf_search_bin_path ../nsg/build/tests/test_nsg_optimized_inf_search --dataset Deep10M --max_cores 16 --construct_R 64 --search_L 64 --omp 1 --batch_size 16
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_nsg_Deep10M_batch_16
python run_all_nsg_inf_search.py --nsg_inf_search_bin_path ../nsg/build/tests/test_nsg_optimized_inf_search --dataset Deep10M --max_cores 16 --construct_R 64 --search_L 64 --omp 1 --batch_size 10000
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_nsg_Deep10M_batch_10000

# SPACEV10M
python run_all_nsg_inf_search.py --nsg_inf_search_bin_path ../nsg/build/tests/test_nsg_optimized_inf_search --dataset SPACEV10M --max_cores 16 --construct_R 64 --search_L 64 --omp 1 --batch_size 1
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_nsg_SPACEV10M_batch_1
python run_all_nsg_inf_search.py --nsg_inf_search_bin_path ../nsg/build/tests/test_nsg_optimized_inf_search --dataset SPACEV10M --max_cores 16 --construct_R 64 --search_L 64 --omp 1 --batch_size 16
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_nsg_SPACEV10M_batch_16
python run_all_nsg_inf_search.py --nsg_inf_search_bin_path ../nsg/build/tests/test_nsg_optimized_inf_search --dataset SPACEV10M --max_cores 16 --construct_R 64 --search_L 64 --omp 1 --batch_size 10000
/usr/lib/linux-tools-5.15.0-101/turbostat --S --interval 1  > log_energy_cpu_nsg_SPACEV10M_batch_10000
```