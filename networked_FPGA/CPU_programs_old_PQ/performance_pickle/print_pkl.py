"""
Example usage:
    python print_pkl.py --fname ./local_network_test_1_FPGA.pkl
"""

import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str, default='./local_network_test_1_FPGA.pkl')

args = parser.parse_args()
fname = args.fname

def load_obj(dirc, name):
    if '.pkl' not in name:
        name = name + '.pkl'
    with open(os.path.join(dirc, name), 'rb') as f:
        return pickle.load(f) 

perf_dict = load_obj('', 'local_network_test_1_FPGA')
print(perf_dict)