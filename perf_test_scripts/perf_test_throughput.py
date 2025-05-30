"""
Test the ANN search throughput throughput on various cases.

(graph_type, dataset, max_degree})	--- these three parameters identifies a unique graph

Example Usage:
# SIFT
python perf_test_throughput.py \
--max_cand_per_group 4 --max_group_num_in_pipe 6 --save_df FPGA_inter_query_4_chan.pickle \
--FPGA_project_dir /mnt/scratch/wenqi/tmp_bitstreams/FPGA_inter_query_v1.3_4_chan_D_128 \
--graph_type HNSW --dataset SIFT1M --max_degree 64 --min_ef 64 --max_ef 64

# Sbert
python perf_test_throughput.py \
--max_cand_per_group 4 --max_group_num_in_pipe 6 --save_df FPGA_inter_query_4_chan.pickle \
--FPGA_project_dir /mnt/scratch/wenqi/tmp_bitstreams/FPGA_inter_query_v1.3_4_chan_D_384 \
--graph_type HNSW --dataset SBERT1M --max_degree 64 --min_ef 64 --max_ef 64
"""

import os
import re
import numpy as np
import argparse 
import pandas as pd

from utils import assert_keywords_in_file, get_number_file_with_keywords, get_FPGA_summary_time

parser = argparse.ArgumentParser()
parser.add_argument('--FPGA_project_dir', type=str, default='/mnt/scratch/wenqi/tmp_bitstreams/FPGA_single_DDR_single_layer_v1.1_async_more_debug')
parser.add_argument('--FPGA_host_name', type=str, default='host', help="the name of the exe of the FPGA host")
parser.add_argument('--FPGA_bin_name', type=str, default='xclbin/vadd.hw.xclbin', help="the name (as well as the subdir) of the FPGA bitstream")
parser.add_argument('--FPGA_log_name', type=str, default='summary.csv', help="the name of the FPGA perf summary file")
parser.add_argument('--max_cand_per_group', type=int, default=4, help="max batch size per iter")
parser.add_argument('--max_group_num_in_pipe', type=int, default=6, help="max number of stages on the fly")
parser.add_argument('--graph_type', type=str, default="HNSW", help="HNSW or NSG")
parser.add_argument('--dataset', type=str, default="SIFT1M", help="SIFT1M, SIFT10M, SBERT1M, etc.")
parser.add_argument('--max_degree', type=int, default=64, help="max degree of the graph (base layer for HNSW)")
parser.add_argument('--min_ef', type=int, default=64, help="max result queue size")
parser.add_argument('--max_ef', type=int, default=64, help="max result queue size")
parser.add_argument('--save_df', type=str, default="perf_df.pickle", help="the performance pickle file to save the dataframe")

args = parser.parse_args()
FPGA_project_dir = args.FPGA_project_dir
FPGA_host_name = args.FPGA_host_name
FPGA_bin_name = args.FPGA_bin_name
FPGA_log_name = args.FPGA_log_name
max_cand_batch_size_in = args.max_cand_per_group
max_async_stage_num_in = args.max_group_num_in_pipe
graph_type = args.graph_type
dataset = args.dataset
max_degree = args.max_degree
ef_list = [] 
# only consider 2^n
for i in range(args.min_ef, args.max_ef + 1):
	if i & (i - 1) == 0:
		ef_list.append(i)
print("ef list:", ef_list)



if __name__ == '__main__':

	# generate a random log name
	log_FPGA = 'log_FPGA' + str(np.random.randint(0, 1000))

	key_columns = ['graph_type', 'dataset', 'max_degree', 'ef', 
			'max_cand_per_group', 'max_group_num_in_pipe']
	result_columns = ['time_ms_kernel', 'recall_1', 'recall_10', 'avg_hops', 'avg_visited']
	columns= key_columns + result_columns

	if os.path.exists(args.save_df): # load existing
		df = pd.read_pickle(args.save_df)
		assert len(df.columns.values) == len(columns)
		for col in columns:
			assert col in df.columns.values
	else:
		# create a dataframe, with primary key being (max_cand_per_group, max_group_num_in_pipe) both int
		#   and columns being (time_ms_kernel, recall_1, recall_10) all float
		#   and debug signals including: avg_hops, avg_visited
		df = pd.DataFrame(columns=columns)
	pd.set_option('display.expand_frame_repr', False) # print all columns

	for ef in ef_list:
		for max_cand_per_group in range(1, max_cand_batch_size_in + 1):
			for max_group_num_in_pipe in range(1, max_async_stage_num_in + 1):
				"""
				Usage: ./host <xclbin> <max_cand_per_group (mc)> <max_group_num_in_pipe (mg)> <ef> <graph_type> <dataset> <Max degree (MD)>
				"""
				host_full = os.path.join(FPGA_project_dir, FPGA_host_name)
				xclbin_full = os.path.join(FPGA_project_dir, FPGA_bin_name)
				cmd_FPGA = f"{host_full} {xclbin_full} {max_cand_per_group} {max_group_num_in_pipe} {ef} {graph_type} {dataset} {max_degree} > {log_FPGA}"
				print("Executing FPGA command:\n", cmd_FPGA)
				os.system(cmd_FPGA)
				
				# assert assert_keywords_in_file(log_FPGA, "Result correct!") == True
				recall_1 = get_number_file_with_keywords(log_FPGA, "Recall@1=", "float")
				recall_10 = get_number_file_with_keywords(log_FPGA, "Recall@10=", "float")
				avg_hops = get_number_file_with_keywords(log_FPGA, "Average #hops on base layer=", "float", allow_none=True)
				avg_visited = get_number_file_with_keywords(log_FPGA, "Average #visited nodes=", "float", allow_none=True)

				# summary will be generated in the current directory
				time_ms_kernel = get_FPGA_summary_time(FPGA_log_name)
				
				print("max_cand_per_group: {}, max_group_num_in_pipe: {}".format(max_cand_per_group, max_group_num_in_pipe))
				print("FPGA end-to-end: {} ms".format(time_ms_kernel))
				print("Recall@1: {}\nRecall@10: {}".format(recall_1, recall_10))
				print("")

				# if already in the df, delete the old row first
				keys_values = {'graph_type': graph_type, 'dataset': dataset, 'max_degree': max_degree, 'ef': ef, 
				   				'max_cand_per_group': max_cand_per_group, 'max_group_num_in_pipe': max_group_num_in_pipe}
				if len(df) > 0:
					idx = df.index[(df['graph_type'] == graph_type) & (df['dataset'] == dataset) & (df['max_degree'] == max_degree) & (df['ef'] == ef) & 
									(df['max_cand_per_group'] == max_cand_per_group) & (df['max_group_num_in_pipe'] == max_group_num_in_pipe)]
					if len(idx) > 0:
						print("Drop the old performance df rows", df.loc[idx])
						df = df.drop(idx)
				# append results
				df = df.append({**keys_values, 'time_ms_kernel': time_ms_kernel, 'recall_1': recall_1, 'recall_10': recall_10, 
								'avg_hops': avg_hops, 'avg_visited': avg_visited}, ignore_index=True)

			
		# print the dataframe of input keys and results, and the best performance
		print("Performance DataFrame:")
		df_added = df.loc[(df['graph_type'] == graph_type) & (df['dataset'] == dataset) & (df['max_degree'] == max_degree) & (df['ef'] == ef)]
		print(df_added)
		
		# show the row with best performance (min kernel time)
		best_time = df_added['time_ms_kernel'].min()
		best_row = df_added.loc[df_added['time_ms_kernel'] == best_time]
		print("Best performance: (ef={})".format(ef))
		print(best_row)

	# save 
	if args.save_df is not None:
		df.to_pickle(args.save_df, protocol=4)
	
	# remove log
	os.system(f"rm {log_FPGA}")
	os.system("rm *csv")