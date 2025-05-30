"""
This scripts executes the CPU index server / CPU coordinator / FPGA simulator programs by reading the config file and automatically generate the command line arguments.

Example 1: one FPGA simulator (on CPU) and one CPU:
	In terminal 1 (run host):
		# real client with dataset
			python launch_CPU_and_FPGA.py --config_fname ./config/local_network_test_1_FPGA.yaml --mode CPU_client
		# simulator (no real dataset)
			python launch_CPU_and_FPGA.py --config_fname ./config/local_network_test_1_FPGA.yaml --mode CPU_client_simulator # --ef 100 --query_num 10000 --batch_size 32 
	In terminal 2:
		python launch_CPU_and_FPGA.py --config_fname ./config/local_network_test_1_FPGA.yaml --mode FPGA_simulator --fpga_id 0 # --ef 100 --query_num 10000 --batch_size 32 

Example 2: one real FPGA and one CPU:
	In terminal 1 (run host):
		# real client with dataset
			python launch_CPU_and_FPGA.py --config_fname ./config/test_1_FPGA.yaml --mode CPU_client
		# simulator (no real dataset)
			python launch_CPU_and_FPGA.py --config_fname ./config/test_1_FPGA.yaml --mode CPU_client_simulator 
	In terminal 2:
		type in the commands output by the first terminal
"""

import argparse 
import json
import os
import yaml
import pandas as pd
pd.set_option('display.expand_frame_repr', False) # print all columns

import numpy as np

from helper import save_obj, load_obj

parser = argparse.ArgumentParser()
parser.add_argument('--config_fname', type=str, default='./config/local_network_test_1_FPGA.yaml')
parser.add_argument('--mode', type=str, help='CPU_client_simulator or FPGA_simulator')

# if run in the CPU mode
parser.add_argument('--cpu_client_exe_dir', type=str, default='./CPU_client', help="the CPP exe file")
parser.add_argument('--cpu_client_simulator_exe_dir', type=str, default='./CPU_client_simulator', help="the CPP exe file")

# if run in the FPGA simulator mode
parser.add_argument('--fpga_simulator_exe_dir', type=str, default='./FPGA_simulator', help="the FPGA simulator exe file")
parser.add_argument('--fpga_id', type=int, default=0, help="the FPGA ID (should start from 0, 1, 2, ...)")

# optional runtim input arguments
parser.add_argument('--ef', type=int, default=None)
parser.add_argument('--query_num', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--max_cand_per_group', type=int, default=None)
parser.add_argument('--max_group_num_in_pipe', type=int, default=None)
parser.add_argument('--graph_type', type=str, default=None)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--max_degree', type=int, default=None)
					

args = parser.parse_args()
config_fname = args.config_fname
mode = args.mode
if mode == 'CPU_client':
	cpu_client_exe_dir = args.cpu_client_exe_dir
elif mode == 'CPU_client_simulator':
	cpu_client_simulator_exe_dir = args.cpu_client_simulator_exe_dir
elif mode == 'FPGA_simulator':
	fpga_simulator_exe_dir = args.fpga_simulator_exe_dir
	fpga_id = args.fpga_id
	
def get_board_ID(FPGA_IP_addr_list):
	"""
	Given the IP address, return the boardNum argument passed to the FPGA network stack
	#   alveo-u250-01: 10.253.74.12
	#   alveo-u250-02: 10.253.74.16
	#   alveo-u250-03: 10.253.74.20
	#   alveo-u250-04: 10.253.74.24
	#   alveo-u250-05: 10.253.74.28
	#   alveo-u250-06: 10.253.74.40
	"""
	if FPGA_IP_addr_list == '10.253.74.12':
		return '1'
	elif FPGA_IP_addr_list == '10.253.74.16':
		return '2'
	elif FPGA_IP_addr_list == '10.253.74.20':
		return '3'
	elif FPGA_IP_addr_list == '10.253.74.24':
		return '4'
	elif FPGA_IP_addr_list == '10.253.74.28':
		return '5'
	elif FPGA_IP_addr_list == '10.253.74.40':
		return '6'
	else:
		return None

# yaml inputs
num_FPGA = None
CPU_IP_addr = None
FPGA_IP_addr_list = None
C2F_port_list = None
F2C_port_list = None

D = None 
ef = None

query_num = None
batch_size = None

query_window_size = None
batch_window_size = None

performance_profile_dir = None
max_cand_per_group = None
max_group_num_in_pipe = None
graph_type = None
dataset = None
max_degree = None

config_dict = {}
with open(args.config_fname, "r") as f:
    config_dict.update(yaml.safe_load(f))
locals().update(config_dict)

if args.ef is not None:
	ef = args.ef
if args.batch_size is not None:
	batch_size = args.batch_size
if args.query_num is not None:
	query_num = args.query_num
if args.max_cand_per_group is not None:
	max_cand_per_group = args.max_cand_per_group
if args.max_group_num_in_pipe is not None:
	max_group_num_in_pipe = args.max_group_num_in_pipe
if args.graph_type is not None:
	graph_type = args.graph_type
if args.dataset is not None:
	dataset = args.dataset
if args.max_degree is not None:
	max_degree = args.max_degree

if dataset is not None:
	if dataset.startswith('SIFT'):
		D = 128
	elif dataset.startswith('Deep'):
		D = 96
	elif dataset.startswith('SBERT'):
		D = 384
	elif dataset.startswith('SPACEV'):
		D = 100
	else:
		raise NotImplementedError

assert int(num_FPGA) == len(FPGA_IP_addr_list)
assert int(num_FPGA) == len(C2F_port_list)
assert int(num_FPGA) == len(F2C_port_list)

# if query_window_size == 'auto':
# 	query_msg_size = 4 * D * (nprobe + 1) # as an approximation


if mode == 'CPU_client':
	"""
  std::cout << "Usage: " << argv[0] << " <1 num_FPGA> "
      "<2 ~ 2 + num_FPGA - 1 FPGA_IP_addr> " 
    "<2 + num_FPGA ~ 2 + 2 * num_FPGA - 1 C2F_port> " 
    "<2 + 2 * num_FPGA ~ 2 + 3 * num_FPGA - 1 F2C_port> "
    "<2 + 3 * num_FPGA dataset> <3 + 3 * num_FPGA graph_type> <4 + 3 * num_FPGA max_degree> <5 + 3 * num_FPGA ef> " 
    "<6 + 3 * num_FPGA query_num> " "<7 + 3 * num_FPGA batch_size> "
    "<8 + 3 * num_FPGA query_window_size> <9 + 3 * num_FPGA batch_window_size> " 
	"""

	assert dataset is not None
	assert graph_type is not None

	if max_cand_per_group == "auto" or max_group_num_in_pipe == "auto":

		assert os.path.exists(performance_profile_dir) # load existing performance
		df = pd.read_pickle(performance_profile_dir)
		df = df.loc[(df['graph_type'] == graph_type) & (df['dataset'] == dataset) & (df['max_degree'] == max_degree) & (df['ef'] == ef)]
		# get the row with max performance (min time_ms_kernel)
		row = df.loc[df['time_ms_kernel'].idxmin()]
		print("Best performance: ", row)
		max_cand_per_group = row['max_cand_per_group']
		max_group_num_in_pipe = row['max_group_num_in_pipe']
		print("Overwriting max_cand_per_group and max_group_num_in_pipe with the best performance: ", 
			max_cand_per_group, max_group_num_in_pipe)
		

	# print the FPGA commands if the FPGA is available
	for i in range(int(num_FPGA)):
		FPGA_IP_addr = FPGA_IP_addr_list[i]	
		if get_board_ID(FPGA_IP_addr) is not None:
			board_ID = get_board_ID(FPGA_IP_addr)
			# std::cout << "Usage: " << argv[0] << " <XCLBIN File 1> <local_FPGA_IP 2> <RxPort (C2F) 3> <TxIP (CPU IP) 4> <TxPort (F2C) 5> <FPGA_board_ID 6" << std::endl;
			print(f'\nnetwork_sim_graph FPGA {i} commands: ')
			print("./host/host build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/network.xclbin "
				f" {FPGA_IP_addr} {C2F_port_list[i]} {CPU_IP_addr} {F2C_port_list[i]} {ef} \n")
			print(f'FPGA_inter_query_v1_3 / FPGA_intra_query_v1_5 FPGA {i} commands: ')
			print("./host/host build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/network.xclbin "
		 		f" {FPGA_IP_addr} {C2F_port_list[i]} {CPU_IP_addr} {F2C_port_list[i]} "
				f"{max_cand_per_group} {max_group_num_in_pipe} {ef} {graph_type} {dataset} {max_degree} {batch_size} \n")
		else:
			print("Unknown FPGA IP address: ", FPGA_IP_addr)

	# execute the CPU command
	cmd = ''
	# cmd += 'taskset --cpu-list 0-{} '.format(cpu_cores)
	cmd += ' {} '.format(cpu_client_exe_dir)
	cmd += ' {} '.format(num_FPGA)
	for i in range(int(num_FPGA)):
		cmd += ' {} '.format(FPGA_IP_addr_list[i])
	for i in range(int(num_FPGA)):
		cmd += ' {} '.format(C2F_port_list[i])
	for i in range(int(num_FPGA)):
		cmd += ' {} '.format(F2C_port_list[i])
	cmd += ' {} '.format(dataset)
	cmd += ' {} '.format(graph_type)
	cmd += ' {} '.format(max_degree)
	cmd += ' {} '.format(ef)
	cmd += ' {} '.format(query_num)
	cmd += ' {} '.format(batch_size)
	cmd += ' {} '.format(query_window_size)
	cmd += ' {} '.format(batch_window_size)
	# cmd += ' {} '.format(cpu_cores)
	print('Executing: ', cmd)
	os.system(cmd)

	# print('Loading and copying profile...')
	latency_ms_distribution = np.fromfile(
		f'latency_ms_per_batch_{dataset}_{graph_type}_MD{max_degree}_ef{ef}_batch_size{batch_size}.double', dtype=np.float64).reshape(-1,)
	# deep copy latency_ms_distribution
	latency_ms_distribution_sorted = np.array(latency_ms_distribution)
	latency_ms_distribution_sorted.sort()
	latency_ms_min = latency_ms_distribution_sorted[0]
	latency_ms_max = latency_ms_distribution_sorted[-1]
	latency_ms_median = latency_ms_distribution_sorted[int(len(latency_ms_distribution_sorted) / 2)]
	# QPS = np.fromfile('profile_QPS.double', dtype=np.float64).reshape(-1,)[0]

	print("Loaded profile: ")
	print("latency_ms_min: ", latency_ms_min)
	print("latency_ms_max: ", latency_ms_max)
	print("latency_ms_median: ", latency_ms_median)
	# print("QPS: ", QPS)
	
	# config_dict['latency_ms_distribution'] = latency_ms_distribution
	# config_dict['latency_ms_min'] = latency_ms_min
	# config_dict['latency_ms_max'] = latency_ms_max
	# config_dict['latency_ms_median'] = latency_ms_median
	# config_dict['QPS'] = QPS

	# fname = os.path.basename(config_fname).split('.')[0] 
	# save_obj(config_dict, 'performance_pickle', fname)

elif mode == 'CPU_client_simulator':
	"""
  std::cout << "Usage: " << argv[0] << " <1 num_FPGA> "
      "<2 ~ 2 + num_FPGA - 1 FPGA_IP_addr> " 
    "<2 + num_FPGA ~ 2 + 2 * num_FPGA - 1 C2F_port> " 
    "<2 + 2 * num_FPGA ~ 2 + 3 * num_FPGA - 1 F2C_port> "
    "<2 + 3 * num_FPGA D> <3 + 3 * num_FPGA ef> " 
    "<4 + 3 * num_FPGA query_num> " "<5 + 3 * num_FPGA batch_size> "
    "<6 + 3 * num_FPGA query_window_size> <7 + 3 * num_FPGA batch_window_size> " 
	"""

	if max_cand_per_group == "auto" or max_group_num_in_pipe == "auto":

		assert os.path.exists(performance_profile_dir) # load existing performance
		df = pd.read_pickle(performance_profile_dir)
		df = df.loc[(df['graph_type'] == graph_type) & (df['dataset'] == dataset) & (df['max_degree'] == max_degree) & (df['ef'] == ef)]
		# get the row with max performance (min time_ms_kernel)
		row = df.loc[df['time_ms_kernel'].idxmin()]
		print("Best performance: ", row)
		max_cand_per_group = row['max_cand_per_group']
		max_group_num_in_pipe = row['max_group_num_in_pipe']
		print("Overwriting max_cand_per_group and max_group_num_in_pipe with the best performance: ", 
			max_cand_per_group, max_group_num_in_pipe)

	# print the FPGA commands if the FPGA is available
	for i in range(int(num_FPGA)):
		FPGA_IP_addr = FPGA_IP_addr_list[i]	
		if get_board_ID(FPGA_IP_addr) is not None:
			board_ID = get_board_ID(FPGA_IP_addr)
			# std::cout << "Usage: " << argv[0] << " <XCLBIN File 1> <local_FPGA_IP 2> <RxPort (C2F) 3> <TxIP (CPU IP) 4> <TxPort (F2C) 5> <FPGA_board_ID 6" << std::endl;
			print(f'\nnetwork_sim_graph FPGA {i} commands: ')
			print("./host/host build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/network.xclbin "
				f" {FPGA_IP_addr} {C2F_port_list[i]} {CPU_IP_addr} {F2C_port_list[i]} {ef} \n")
			print(f'FPGA_inter_query_v1_3 / FPGA_intra_query_v1_5 FPGA {i} commands: ')
			print("./host/host build_dir.hw.xilinx_u250_gen3x16_xdma_4_1_202210_1/network.xclbin "
		 		f" {FPGA_IP_addr} {C2F_port_list[i]} {CPU_IP_addr} {F2C_port_list[i]} "
				f"{max_cand_per_group} {max_group_num_in_pipe} {ef} {graph_type} {dataset} {max_degree} {batch_size} \n")
		else:
			print("Unknown FPGA IP address: ", FPGA_IP_addr)

	# execute the CPU command
	cmd = ''
	# cmd += 'taskset --cpu-list 0-{} '.format(cpu_cores)
	cmd += ' {} '.format(cpu_client_simulator_exe_dir)
	cmd += ' {} '.format(num_FPGA)
	for i in range(int(num_FPGA)):
		cmd += ' {} '.format(FPGA_IP_addr_list[i])
	for i in range(int(num_FPGA)):
		cmd += ' {} '.format(C2F_port_list[i])
	for i in range(int(num_FPGA)):
		cmd += ' {} '.format(F2C_port_list[i])
	cmd += ' {} '.format(D)
	cmd += ' {} '.format(ef)
	cmd += ' {} '.format(query_num)
	cmd += ' {} '.format(batch_size)
	cmd += ' {} '.format(query_window_size)
	cmd += ' {} '.format(batch_window_size)
	# cmd += ' {} '.format(cpu_cores)
	print('Executing: ', cmd)
	os.system(cmd)

	print('Loading and copying profile...')
	latency_ms_distribution = np.fromfile('profile_latency_ms_distribution.double', dtype=np.float64).reshape(-1,)
	# deep copy latency_ms_distribution
	latency_ms_distribution_sorted = np.array(latency_ms_distribution)
	latency_ms_distribution_sorted.sort()
	latency_ms_min = latency_ms_distribution_sorted[0]
	latency_ms_max = latency_ms_distribution_sorted[-1]
	latency_ms_median = latency_ms_distribution_sorted[int(len(latency_ms_distribution_sorted) / 2)]
	QPS = np.fromfile('profile_QPS.double', dtype=np.float64).reshape(-1,)[0]

	print("Loaded profile: ")
	print("latency_ms_min: ", latency_ms_min)
	print("latency_ms_max: ", latency_ms_max)
	print("latency_ms_median: ", latency_ms_median)
	print("QPS: ", QPS)
	
	# config_dict['latency_ms_distribution'] = latency_ms_distribution
	# config_dict['latency_ms_min'] = latency_ms_min
	# config_dict['latency_ms_max'] = latency_ms_max
	# config_dict['latency_ms_median'] = latency_ms_median
	# config_dict['QPS'] = QPS

	# fname = os.path.basename(config_fname).split('.')[0] 
	# save_obj(config_dict, 'performance_pickle', fname)


elif mode == 'FPGA_simulator':
	"""
// std::cout << "Usage: 
//  " << argv[0] << " <Tx (CPU) IP_addr> <Tx F2C_port> <Rx C2F_port> " 
// 	"<TOPK/ef> <D> <query_num> " << std::endl;
	"""
	cmd = ''
	cmd += ' {} '.format(fpga_simulator_exe_dir)
	cmd += ' {} '.format(CPU_IP_addr)
	cmd += ' {} '.format(F2C_port_list[fpga_id])
	cmd += ' {} '.format(C2F_port_list[fpga_id])
	cmd += ' {} '.format(ef)
	cmd += ' {} '.format(D)
	cmd += ' {} '.format(query_num)
	print('Executing: ', cmd)
	os.system(cmd)