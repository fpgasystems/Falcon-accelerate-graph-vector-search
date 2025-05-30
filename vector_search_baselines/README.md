# CPU and GPU vector search baselines

This repo contains all the vector search baseline experiments for HNSW, NSG, Faiss, GGNN, as well as the index conversion scripts for FPGAs.


**Clone the repository**

```shell
git clone --recursive https://github.com/PomeloTea0726/hnsw_experiments.git
cd hnsw_experiments
```

To update:
```
git pull --recurse-submodules
```

## Setup Environments

### Install libaraies

If using CPU/GPU instances on AWS, choose the DeepLearning Pytorch version (likely Ubuntu 20.04), as it has basic tools installed such as cmake.

Install conda:
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh 
./Anaconda3-2023.09-0-Linux-x86_64.sh 
```

Google perftools for NSG:
```
sudo apt-get install libgoogle-perftools-dev
```

Boost for NSG:
```
sudo apt-get install libboost-all-dev
```

turbostat for energy consumption measurement on CPU (this does not work on VMs):
	be aware that installing this can cause apt to break --- cannot install other packages! So install this last
```
# turbostat
# Command 'turbostat' not found, but can be installed with:
# sudo apt install linux-intel-iotg-tools-common        # version 5.15.0-1043.49, or
# sudo apt install linux-nvidia-6.2-tools-common        # version 6.2.0-1003.3~22.04.1

sudo apt install linux-intel-iotg-tools-common

# non-root users can be enabled to run turbostat this way: 
# sudo setcap cap_sys_rawio=ep /usr/lib/linux-aws-6.5-tools-6.5.0-1020/turbostat
# sudo chmod +r /dev/cpu/*/msr 
# sudo chmod +r /dev/cpu_dma_latency

# if using non-root user does not work, just use it with sudo:
sudo /usr/lib/linux-aws-6.5-tools-6.5.0-1020/turbostat --S --interval 1 
```

### NSG

**Compile nsg**
```shell
mkdir nsg/build
cd nsg/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j16
```

See `scripts_nsg/README.md`

### HNSW

**Compile hnswlib**

if need to use graph construction
```
pip3 install hnswlib
```

```
mkdir hnswlib/build
cd hnswlib/build
cmake ..
make 
```

See `scripts_hnsw/README.md`

### Faiss

```
conda create -n py39 python=3.9 -y
conda activate py39

# GPU version:
# install CUDA first if its not installed: https://docs.vmware.com/en/VMware-vSphere-Bitfusion/3.0/Example-Guide/GUID-ABB4A0B1-F26E-422E-85C5-BA9F2454363A.html
conda install faiss-gpu=1.7.3 -c conda-forge
pip3 install pandas==1.3.5

# cpu version (included in GPU version, no need to install again)
# conda install -c conda-forge openblas
# conda install -c pytorch faiss-cpu==1.7.3
```

## TODO

- [ ] 
