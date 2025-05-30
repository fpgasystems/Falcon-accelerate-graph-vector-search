import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns 

# Seaborn version >= 0.11.0

sns.set_theme(style="whitegrid")
# Set the palette to the "pastel" default palette:
# sns.set_palette("pastel")

def plot_latency(dataset='SIFT1M', graph_type="HNSW", max_degree=64, ef=64, batch_sizes=[1,2,4,8,16,32]):

	### Note: For violin graph, a single violin's data must be in the same column
	###   e.g., given 3 violin plots, each with 100 points, the shape of the array
	###   should be (100, 3), where the first column is for the first violin and so forth
	# fake up some data


	# Wenqi: flatten the table to a table. It's a dictionary with the key as schema.
	#   The value of each key is an array.
	# label category data
	# xxx   xxx      xxx
	# yyy   yyy      yyy

	folder_name_inter_query = 'saved_latency/FPGA_inter_query_v1_3_4_chan'
	folder_name_intra_query = 'saved_latency/FPGA_intra_query_v1_5_4_chan'

	d = {}
	d['label'] = []
	d['data'] = []
	d['category'] = []

	for batch_size in batch_sizes:
		# load latency distribution (in double)
		# file name
		#   std::string out_fname = "latency_ms_per_batch_" + dataset + "_" + graph_type + 
		# 	  "_MD" + std::to_string(max_degree) + "_ef" + std::to_string(ef) + + "_batch_size" + std::to_string(batch_size) + ".double";
		f_name_inter_query = os.path.join(folder_name_inter_query, 
			"latency_ms_per_batch_" + dataset + "_" + graph_type + "_MD" + str(max_degree) + "_ef" + str(ef) + "_batch_size" + str(batch_size) + ".double")
		f_name_intra_query = os.path.join(folder_name_intra_query,
			"latency_ms_per_batch_" + dataset + "_" + graph_type + "_MD" + str(max_degree) + "_ef" + str(ef) + "_batch_size" + str(batch_size) + ".double")
		# load as np array
		latency_inter_query = np.fromfile(f_name_inter_query, dtype=np.float64)
		latency_intra_query = np.fromfile(f_name_intra_query, dtype=np.float64)

		for latency in latency_inter_query:
			d['label'].append('batch_size={}'.format(batch_size))
			d['data'].append(latency)
			d['category'].append('Falcon Inter-query Parallel')
		
		for latency in latency_intra_query:
			d['label'].append('batch_size={}'.format(batch_size))
			d['data'].append(latency)
			d['category'].append('Falcon Intra-query Parallel')

	df = pd.DataFrame(data=d)
	print(df.index)
	print(df.columns)

	plt.figure(figsize=(6, 3))
	# API: https://seaborn.pydata.org/generated/seaborn.violinplot.html
	# inner{“box”, “quartile”, “point”, “stick”, None}, optional
	# ax = sns.violinplot(data=df, scale='area', inner='box', x="label", y="data", hue="category")
	# use box plot
	ax = sns.boxplot(data=df, x="label", y="data", hue="category", showfliers=False)

	x_tick_labels = ["{}".format(i) for i in batch_sizes]
	ax.set_xticklabels(x_tick_labels)
	# ax.set_yticklabels(y_tick_labels)
	# plt.yticks(rotation=0)
	# # ax.ticklabel_format(axis='both', style='sci')
	# # ax.set_yscale("log")
	ax.legend(loc='upper left')

	ax.tick_params(length=0, top=False, bottom=False, left=False, right=False, 
		labelleft=True, labelbottom=True, labelsize=12)
	ax.set_xlabel('Batch sizes', fontsize=12, labelpad=10)
	ax.set_ylabel('Latency per batch (ms)', fontsize=12, labelpad=10)
	ax.set_title(f'Dataset: {dataset}, Graph: {graph_type}', fontsize=12)
	# plt.text(2, len(y_tick_labels) + 2, "Linear Heatmap", fontsize=16)

	plt.savefig(f'./images/latency_FPGA/latency_FPGA_{dataset}_{graph_type}.png', transparent=False, dpi=200, bbox_inches="tight")

	# plt.show()

if __name__ == "__main__":

	datasets = ["SIFT1M", "SIFT10M", "Deep1M", "Deep10M", "SPACEV1M", "SPACEV10M"]
	graph_types = ["HNSW", "NSG"]
	batch_sizes=[1,2,4,8,16,32]

	for dataset in datasets:
		for graph_type in graph_types:
			plot_latency(dataset, graph_type,batch_sizes=batch_sizes)