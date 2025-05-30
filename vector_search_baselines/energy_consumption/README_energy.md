# Energy Measurement

The specific instructions for energy measurement are included in the instructions of the README of each libraray scripts. Here are the instructions for reporting the energy (turbostat for CPU and nvidia-smi for GPU)

CPU (r630): 
```
python turbostat_energy_parsing.py --fname log_CPU_r630

File: log_energy_cpu_faiss_Deep10M_batch_1      Average energy consumption: 75.62 W
File: log_energy_cpu_faiss_Deep10M_batch_10000  Average energy consumption: 109.45 W
File: log_energy_cpu_faiss_Deep10M_batch_16     Average energy consumption: 128.98 W
File: log_energy_cpu_faiss_SIFT10M_batch_1      Average energy consumption: 75.56 W
File: log_energy_cpu_faiss_SIFT10M_batch_10000  Average energy consumption: 105.22 W
File: log_energy_cpu_faiss_SIFT10M_batch_16     Average energy consumption: 129.99 W
File: log_energy_cpu_faiss_SPACEV10M_batch_1    Average energy consumption: 78.73 W
File: log_energy_cpu_faiss_SPACEV10M_batch_10000        Average energy consumption: 105.02 W
File: log_energy_cpu_faiss_SPACEV10M_batch_16   Average energy consumption: 130.40 W
File: log_energy_cpu_hnsw_Deep10M_batch_1       Average energy consumption: 72.43 W
File: log_energy_cpu_hnsw_Deep10M_batch_10000   Average energy consumption: 110.57 W
File: log_energy_cpu_hnsw_Deep10M_batch_16      Average energy consumption: 106.68 W
File: log_energy_cpu_hnsw_SIFT10M_batch_1       Average energy consumption: 66.95 W
File: log_energy_cpu_hnsw_SIFT10M_batch_10000   Average energy consumption: 109.46 W
File: log_energy_cpu_hnsw_SIFT10M_batch_16      Average energy consumption: 105.88 W
File: log_energy_cpu_hnsw_SPACEV10M_batch_1     Average energy consumption: 66.83 W
File: log_energy_cpu_hnsw_SPACEV10M_batch_10000 Average energy consumption: 106.49 W
File: log_energy_cpu_hnsw_SPACEV10M_batch_16    Average energy consumption: 100.38 W
File: log_energy_cpu_nsg_Deep10M_batch_1        Average energy consumption: 66.46 W
File: log_energy_cpu_nsg_Deep10M_batch_10000    Average energy consumption: 110.55 W
File: log_energy_cpu_nsg_Deep10M_batch_16       Average energy consumption: 106.28 W
File: log_energy_cpu_nsg_SIFT10M_batch_1        Average energy consumption: 66.59 W
File: log_energy_cpu_nsg_SIFT10M_batch_10000    Average energy consumption: 111.25 W
File: log_energy_cpu_nsg_SIFT10M_batch_16       Average energy consumption: 104.23 W
File: log_energy_cpu_nsg_SPACEV10M_batch_1      Average energy consumption: 66.75 W
File: log_energy_cpu_nsg_SPACEV10M_batch_10000  Average energy consumption: 109.99 W
File: log_energy_cpu_nsg_SPACEV10M_batch_16     Average energy consumption: 104.74 W
File: log_idle  Average energy consumption: 63.16 W
```

CPU (m5.metal 48 cores, 10000 batch size using 16/48 cores): 
	the idle energy can be used as a deduction for estimating real 16-core energy setup
```
python turbostat_energy_parsing.py --fname log_CPU_m5.metal_use_48_cores/

File: log_energy_cpu_faiss_Deep10M_batch_1      Average energy consumption: 140.49 W
File: log_energy_cpu_faiss_Deep10M_batch_10000  Average energy consumption: 305.73 W
File: log_energy_cpu_faiss_Deep10M_batch_10000_16_cores Average energy consumption: 209.85 W
File: log_energy_cpu_faiss_Deep10M_batch_16     Average energy consumption: 217.16 W
File: log_energy_cpu_faiss_SIFT10M_batch_1      Average energy consumption: 139.20 W
File: log_energy_cpu_faiss_SIFT10M_batch_10000  Average energy consumption: 306.71 W
File: log_energy_cpu_faiss_SIFT10M_batch_10000_16_cores Average energy consumption: 209.70 W
File: log_energy_cpu_faiss_SIFT10M_batch_16     Average energy consumption: 209.53 W
File: log_energy_cpu_faiss_SPACEV10M_batch_1    Average energy consumption: 140.03 W
File: log_energy_cpu_faiss_SPACEV10M_batch_10000        Average energy consumption: 305.53 W
File: log_energy_cpu_faiss_SPACEV10M_batch_10000_16_cores       Average energy consumption: 206.88 W
File: log_energy_cpu_faiss_SPACEV10M_batch_16   Average energy consumption: 215.35 W
File: log_energy_cpu_hnsw_Deep10M_batch_1       Average energy consumption: 137.97 W
File: log_energy_cpu_hnsw_Deep10M_batch_10000   Average energy consumption: 322.40 W
File: log_energy_cpu_hnsw_Deep10M_batch_10000_16_cores  Average energy consumption: 212.18 W
File: log_energy_cpu_hnsw_Deep10M_batch_16      Average energy consumption: 202.43 W
File: log_energy_cpu_hnsw_SIFT10M_batch_1       Average energy consumption: 136.93 W
File: log_energy_cpu_hnsw_SIFT10M_batch_10000   Average energy consumption: 314.01 W
File: log_energy_cpu_hnsw_SIFT10M_batch_10000_16_cores  Average energy consumption: 212.96 W
File: log_energy_cpu_hnsw_SIFT10M_batch_16      Average energy consumption: 201.61 W
File: log_energy_cpu_hnsw_SPACEV10M_batch_1     Average energy consumption: 138.55 W
File: log_energy_cpu_hnsw_SPACEV10M_batch_10000 Average energy consumption: 344.51 W
File: log_energy_cpu_hnsw_SPACEV10M_batch_10000_16_cores        Average energy consumption: 220.81 W
File: log_energy_cpu_hnsw_SPACEV10M_batch_16    Average energy consumption: 209.26 W
File: log_energy_cpu_nsg_Deep10M_batch_1        Average energy consumption: 136.71 W
File: log_energy_cpu_nsg_Deep10M_batch_10000    Average energy consumption: 321.89 W
File: log_energy_cpu_nsg_Deep10M_batch_10000_16_cores   Average energy consumption: 203.40 W
File: log_energy_cpu_nsg_Deep10M_batch_16       Average energy consumption: 190.78 W
File: log_energy_cpu_nsg_SIFT10M_batch_1        Average energy consumption: 136.51 W
File: log_energy_cpu_nsg_SIFT10M_batch_10000    Average energy consumption: 308.46 W
File: log_energy_cpu_nsg_SIFT10M_batch_10000_16_cores   Average energy consumption: 206.51 W
File: log_energy_cpu_nsg_SIFT10M_batch_16       Average energy consumption: 192.94 W
File: log_energy_cpu_nsg_SPACEV10M_batch_1      Average energy consumption: 136.49 W
File: log_energy_cpu_nsg_SPACEV10M_batch_10000  Average energy consumption: 319.15 W
File: log_energy_cpu_nsg_SPACEV10M_batch_10000_16_cores Average energy consumption: 204.92 W
File: log_energy_cpu_nsg_SPACEV10M_batch_16     Average energy consumption: 192.46 W
File: log_idle  Average energy consumption: 90.60 W
```

GPU (3090): 

```
python compute_GPU_average_energy.py --fname log_GPU_3090/

File: log_GPU_3090/log_energy_gpu_faiss_Deep10M_batch_1 Average energy consumption: 188.59
File: log_GPU_3090/log_energy_gpu_faiss_Deep10M_batch_10000     Average energy consumption: 342.67
File: log_GPU_3090/log_energy_gpu_faiss_Deep10M_batch_16        Average energy consumption: 324.18
File: log_GPU_3090/log_energy_gpu_faiss_SIFT10M_batch_1 Average energy consumption: 183.49
File: log_GPU_3090/log_energy_gpu_faiss_SIFT10M_batch_10000     Average energy consumption: 347.30
File: log_GPU_3090/log_energy_gpu_faiss_SIFT10M_batch_16        Average energy consumption: 270.53
File: log_GPU_3090/log_energy_gpu_faiss_SPACEV10M_batch_1       Average energy consumption: 181.77
File: log_GPU_3090/log_energy_gpu_faiss_SPACEV10M_batch_10000   Average energy consumption: 339.18
File: log_GPU_3090/log_energy_gpu_faiss_SPACEV10M_batch_16      Average energy consumption: 286.72
File: log_GPU_3090/log_energy_gpu_ggnn_Deep10M_batch_1  Average energy consumption: 153.43
File: log_GPU_3090/log_energy_gpu_ggnn_Deep10M_batch_10000      Average energy consumption: 314.70
File: log_GPU_3090/log_energy_gpu_ggnn_Deep10M_batch_16 Average energy consumption: 157.01
File: log_GPU_3090/log_energy_gpu_ggnn_SIFT10M_batch_1  Average energy consumption: 151.29
File: log_GPU_3090/log_energy_gpu_ggnn_SIFT10M_batch_10000      Average energy consumption: 290.84
File: log_GPU_3090/log_energy_gpu_ggnn_SIFT10M_batch_16 Average energy consumption: 156.88
File: log_GPU_3090/log_energy_gpu_ggnn_SPACEV10M_batch_1        Average energy consumption: 155.10
File: log_GPU_3090/log_energy_gpu_ggnn_SPACEV10M_batch_10000    Average energy consumption: 291.52
File: log_GPU_3090/log_energy_gpu_ggnn_SPACEV10M_batch_16       Average energy consumption: 155.59
```

GPU (V100 16GB):

```
python compute_GPU_average_energy.py --fname log_GPU_V100/

File: log_GPU_V100/log_energy_gpu_faiss_Deep10M_batch_1 Average energy consumption: 73.16
File: log_GPU_V100/log_energy_gpu_faiss_Deep10M_batch_10000     Average energy consumption: 208.37
File: log_GPU_V100/log_energy_gpu_faiss_Deep10M_batch_16        Average energy consumption: 153.88
File: log_GPU_V100/log_energy_gpu_faiss_SIFT10M_batch_1 Average energy consumption: 67.26
File: log_GPU_V100/log_energy_gpu_faiss_SIFT10M_batch_10000     Average energy consumption: 168.08
File: log_GPU_V100/log_energy_gpu_faiss_SIFT10M_batch_16        Average energy consumption: 116.24
File: log_GPU_V100/log_energy_gpu_faiss_SPACEV10M_batch_1       Average energy consumption: 66.42
File: log_GPU_V100/log_energy_gpu_faiss_SPACEV10M_batch_10000   Average energy consumption: 176.42
File: log_GPU_V100/log_energy_gpu_faiss_SPACEV10M_batch_16      Average energy consumption: 125.75
File: log_GPU_V100/log_energy_gpu_ggnn_Deep10M_batch_1  Average energy consumption: 51.00
File: log_GPU_V100/log_energy_gpu_ggnn_Deep10M_batch_10000      Average energy consumption: 192.26
File: log_GPU_V100/log_energy_gpu_ggnn_Deep10M_batch_16 Average energy consumption: 52.03
File: log_GPU_V100/log_energy_gpu_ggnn_SIFT10M_batch_1  Average energy consumption: 52.51
File: log_GPU_V100/log_energy_gpu_ggnn_SIFT10M_batch_10000      Average energy consumption: 152.51
File: log_GPU_V100/log_energy_gpu_ggnn_SIFT10M_batch_16 Average energy consumption: 52.98
File: log_GPU_V100/log_energy_gpu_ggnn_SPACEV10M_batch_1        Average energy consumption: 56.00
File: log_GPU_V100/log_energy_gpu_ggnn_SPACEV10M_batch_10000    Average energy consumption: 152.75
File: log_GPU_V100/log_energy_gpu_ggnn_SPACEV10M_batch_16       Average energy consumption: 56.45
```