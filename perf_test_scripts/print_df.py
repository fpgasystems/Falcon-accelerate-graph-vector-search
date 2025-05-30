"""
Print a dataframe loaded from pickle

Example usage:
	python print_df.py --path perf_df.pickle
"""

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='perf_df.pickle', help="the path to the pickle file")
args = parser.parse_args()

pd.set_option('display.expand_frame_repr', False) # print all columns
pd.set_option('display.max_rows', None) # print all rows

# load
df = pd.read_pickle(args.path)
print(df)