# GGNN (GPU) scripts

## Construct GGNN HNSW Indexes

In `run_all_ggnn_construct_and_search.py`, make sure the construction parameters are correct (mainly `construct_KBuild_list` and `construct_S_list` while the other two are irrelevant to construction):
```
    construct_KBuild_list = [32, 64]         # number of neighbors per point in the graph
    construct_S_list = [64]              # segment/batch size (needs to be > KBuild-KF)
    construct_KQuery_list = [10]         # number of neighbors to search for
    construct_MaxIter_list = [400] # number of iterations for BFiS
```

Run construction. Make sure to fill the right GPU ID for servers with multiple GPUs:
```
python run_all_ggnn_construct_and_search.py --mode construct --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SIFT1M --gpu_id 2
python run_all_ggnn_construct_and_search.py --mode construct --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SIFT10M --gpu_id 2
python run_all_ggnn_construct_and_search.py --mode construct --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset Deep1M --gpu_id 2
python run_all_ggnn_construct_and_search.py --mode construct --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset Deep10M --gpu_id 2
python run_all_ggnn_construct_and_search.py --mode construct --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SPACEV1M --gpu_id 2
python run_all_ggnn_construct_and_search.py --mode construct --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SPACEV10M --gpu_id 2
```

## Evaluate GPU latency & throughput

In `run_all_ggnn_construct_and_search.py`, make sure the search parameters are correct:
* search_KBuild_list and search_S_list should be the ones used for construction
* search_KQuery_list can be fixed to 10
* search_tauq_list can be fixed to 0.5
* only need to tune `search_MaxIter_list` (max iteration per query) and `search_bs_list` for batch sizes

```
    search_KBuild_list = [32, 64]           # should be subset of construct_KBuild_list
    search_S_list = [64]                # should be subset of construct_S_list
    search_KQuery_list = [10]
    search_MaxIter_list = [1, 32, 64, 100, 200, 400]
    search_tauq_list = [0.5]
    search_bs_list = [1, 2, 4, 8, 16, 10000]
```

Run search. Make sure to fill the right GPU ID for servers with multiple GPUs:
```
python run_all_ggnn_construct_and_search.py --mode search --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SIFT1M --gpu_id 2 --nruns 3 --perf_df_path perf_df_ggnn_gpu.pickle
python run_all_ggnn_construct_and_search.py --mode search --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SIFT10M --gpu_id 2 --nruns 3 --perf_df_path perf_df_ggnn_gpu.pickle
python run_all_ggnn_construct_and_search.py --mode search --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset Deep1M --gpu_id 2 --nruns 3 --perf_df_path perf_df_ggnn_gpu.pickle
python run_all_ggnn_construct_and_search.py --mode search --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset Deep10M --gpu_id 2 --nruns 3 --perf_df_path perf_df_ggnn_gpu.pickle
python run_all_ggnn_construct_and_search.py --mode search --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SPACEV1M --gpu_id 2 --nruns 3 --perf_df_path perf_df_ggnn_gpu.pickle
python run_all_ggnn_construct_and_search.py --mode search --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SPACEV10M --gpu_id 2 --nruns 3 --perf_df_path perf_df_ggnn_gpu.pickle
```

## Evaluate GPU energy 

Use two terminals, one for executing the program, the other for tracking energy.

```
# SIFT10M
python run_all_ggnn_inf_search.py --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SIFT10M \
        --KBuild 64 --S 64 --KQuery 10 --MaxIter 400 --tauq 0.5 --bs 1 --gpu_id 0 
nvidia-smi -l 1 > log_energy_gpu_ggnn_SIFT10M_batch_1
python run_all_ggnn_inf_search.py --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SIFT10M \
        --KBuild 64 --S 64 --KQuery 10 --MaxIter 400 --tauq 0.5 --bs 16 --gpu_id 0 
nvidia-smi -l 1 > log_energy_gpu_ggnn_SIFT10M_batch_16
python run_all_ggnn_inf_search.py --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SIFT10M \
        --KBuild 64 --S 64 --KQuery 10 --MaxIter 400 --tauq 0.5 --bs 10000 --gpu_id 0 
nvidia-smi -l 1 > log_energy_gpu_ggnn_SIFT10M_batch_10000

# Deep10M
python run_all_ggnn_inf_search.py --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset Deep10M \
        --KBuild 64 --S 64 --KQuery 10 --MaxIter 400 --tauq 0.5 --bs 1 --gpu_id 0 
nvidia-smi -l 1 > log_energy_gpu_ggnn_Deep10M_batch_1
python run_all_ggnn_inf_search.py --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset Deep10M \
        --KBuild 64 --S 64 --KQuery 10 --MaxIter 400 --tauq 0.5 --bs 16 --gpu_id 0 
nvidia-smi -l 1 > log_energy_gpu_ggnn_Deep10M_batch_16
python run_all_ggnn_inf_search.py --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset Deep10M \
        --KBuild 64 --S 64 --KQuery 10 --MaxIter 400 --tauq 0.5 --bs 10000 --gpu_id 0 
nvidia-smi -l 1 > log_energy_gpu_ggnn_Deep10M_batch_10000

# SPACEV10M
python run_all_ggnn_inf_search.py --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SPACEV10M \
        --KBuild 64 --S 64 --KQuery 10 --MaxIter 400 --tauq 0.5 --bs 1 --gpu_id 0 
nvidia-smi -l 1 > log_energy_gpu_ggnn_SPACEV10M_batch_1
python run_all_ggnn_inf_search.py --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SPACEV10M \
        --KBuild 64 --S 64 --KQuery 10 --MaxIter 400 --tauq 0.5 --bs 16 --gpu_id 0 
nvidia-smi -l 1 > log_energy_gpu_ggnn_SPACEV10M_batch_16
python run_all_ggnn_inf_search.py --ggnn_index_path ../data/GPU_GGNN_GRAPH/ --ggnn_bin_path ../ggnn/build_local/ --dataset SPACEV10M \
        --KBuild 64 --S 64 --KQuery 10 --MaxIter 400 --tauq 0.5 --bs 10000 --gpu_id 0 
nvidia-smi -l 1 > log_energy_gpu_ggnn_SPACEV10M_batch_10000
```

## Notes on GGNN

Main ggnn files used for evaluation includes: `sift.cu`, `deep.cu` and `spacev.cu` (see `ggnn/src/`) depending on the targeted dataset. For index construction, GGNN latency and throughput evaluation, see `run_all_ggnn_construct_and_search.py`. For energy evaluation, see `run_all_ggnn_inf_search.py`.

In GGNN construction mode, 2 main parameters are involved as follows. Output index file will be named as `{dataset}_KB{KBuild}_S{S}.bin`.

* `KBuild`: Number of neighbors per point in the graph.

* `S`: Segment/batch size (needs to be > KBuild/2).

In GGNN search mode, 6 main parameters are involved as follows. Note that here `Kbuild` and `S` are used to locate the index file.

* `KBuild`: Number of neighbors per point in the graph (degree).

* `S`: Segment/batch size (needs to be > KBuild/2).

* `KQuery`: Number of neighbors to search for (similar to `L` in NSG and `ef` in HNSW).

* `MaxIter`: Number of iterations for BFiS (hard upper bound).

* `tauq`: Parameter tau for query, balancing between the search performance and recall.

* `bs`: Batch size.

In addition, there are several fixed parameters in search mode which can also be tuned manually (e.g. see `sift.cu`).

* `L`: Graph height / number of layers (4 usually performs best).

* `BLOCK_DIM_X`

* `CACHE_SIZE`

* `SORTED_SIZE`
