import numpy as np
import os
import sys
import re

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help="The GPU ID to measure energy")
parser.add_argument('--verbose', type=int, default=0, help="whether to print the energy per tick")
parser.add_argument('--fname', type=str, default='./out_energy', help="The nvidia-smi file measured by nvidia-smi -l 1 ")

args = parser.parse_args()
gpu_id = args.gpu_id
dir_in = args.fname

"""
Example script:

| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA TITAN X ...  On   | 00000000:04:00.0 Off |                  N/A |
| 39%   74C    P2   226W / 250W |  10853MiB / 12288MiB |     81%      Default |
|     

Another example:

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 520.61.05    Driver Version: 520.61.05    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:16.0 Off |                    0 |
| N/A   46C    P0   178W / 300W |   4285MiB / 32768MiB |     91%      Default |
|                               |                      |                  N/A |

"""

def print_energy(dir_in, gpu_id=0, verbose=False):
    with open(dir_in, 'r') as f:
        file = f.readlines()

    energy_consumption = []
    for i, line in enumerate(file):
        ID_match_str_1 = "{}  Tesla".format(str(gpu_id))
        match_1 = re.search(ID_match_str_1, line) # next line is energy
        ID_match_str_2 = "{}  NVIDIA".format(str(gpu_id))
        match_2 = re.search(ID_match_str_2, line) # next line is energy
        if match_1 or match_2:
            if i + 1 < len(file):
                str_list = re.split(' ', file[i + 1])
                for s in str_list:
                    if re.search('W', s):
                        energy = int(s[:-1])
                        energy_consumption.append(energy)
                        break

    if verbose:
        print(energy_consumption)
        print("Count of frames: {}".format(len(energy_consumption)))
    print("File: {}\tAverage energy consumption: {:.2f}".format(dir_in, np.average(energy_consumption)))

if os.path.isdir(dir_in):
    files = os.listdir(dir_in)
    files.sort()
    for file in files:
        print_energy(os.path.join(dir_in, file), gpu_id, args.verbose)
else:
    print_energy(dir_in, gpu_id, args.verbose)