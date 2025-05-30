"""
Convert a hnsw index to FPGA format

Example Usage:
python hnsw_to_FPGA.py --dbname SIFT1M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT1M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT1M_MD64
python hnsw_to_FPGA.py --dbname SIFT10M --CPU_index_path ../data/CPU_hnsw_indexes/SIFT10M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/SIFT10M_MD64
python hnsw_to_FPGA.py --dbname Deep10M --CPU_index_path ../data/CPU_hnsw_indexes/Deep10M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/Deep10M_MD64
python hnsw_to_FPGA.py --dbname GLOVE --CPU_index_path ../data/CPU_hnsw_indexes/GLOVE_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/GLOVE_MD64
python hnsw_to_FPGA.py --dbname SBERT1M --CPU_index_path ../data/CPU_hnsw_indexes/SBERT1M_index_MD64.bin --FPGA_index_path ../data/FPGA_hnsw/SBERT1M_MD64
"""

import argparse
import time
import numpy as np 
import sys
import os

from hnsw import HNSW_index

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dbname', type=str, help='name of the database, e.g., SIFT10M, Deep10M, GLOVE')
    parser.add_argument('--CPU_index_path', type=str, default='../data/CPU_hnsw_indexes/SIFT1M_index_MD64.bin', help='input path to the hnsw index .bin file')
    parser.add_argument('--FPGA_index_path', type=str, default='../data/FPGA_hnsw/SIFT1M_MD64', help='output path to the FPGA indexes')
    
    args = parser.parse_args()
    assert os.path.exists(args.CPU_index_path), "CPU index file does not exist"

    if "SIFT" in args.dbname:
        dim = 128
    elif "Deep" in args.dbname:
        dim = 96
    elif "GLOVE" in args.dbname:
        dim = 300
    elif "SBERT" in args.dbname:
        dim = 384
    elif "SPACEV" in args.dbname:
        dim = 100
    else:
        raise ValueError("Unsupported database name")

    hnsw_index = HNSW_index(dim=dim)
    hnsw_index.load_index_and_data(args.CPU_index_path)

    if not os.path.exists(args.FPGA_index_path):
        os.makedirs(args.FPGA_index_path)
    byte_array_meta, byte_array_ground_links, byte_array_ground_labels, byte_array_ground_vectors, byte_array_upper_links, byte_array_upper_links_pointers = \
        hnsw_index.save_as_FPGA_format(args.FPGA_index_path, num_channels=[1, 2, 4])