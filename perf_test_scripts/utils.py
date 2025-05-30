import os
import re

def assert_keywords_in_file(fname, keyword):
	"""
	Given an input text file, assert there is at least a line that contains a keyword
	"""
	with open(fname) as f:
		lines = f.readlines()
		for line in lines:
			if keyword in line:
				return True
	return False

def get_number_file_with_keywords(fname, keyword, dtype='int', allow_none=False):
	"""
	Given an input text file, find the line that contains a keyword,
		and extract the first number (int or float)
	"""
	assert dtype == 'int' or dtype == 'float'
	if dtype == 'int':
		pattern = r"\d+"
	elif dtype == 'float':
		pattern = r"(\d+.\d+)"

	result = None
	with open(fname) as f:
		lines = f.readlines()
		for line in lines:
			if keyword in line:
				# remove the keyword
				line = line.replace(keyword, "")
				result = re.findall(pattern, line)[0]
	if not allow_none:
		assert result is not None
	elif result is None:
		return None
	if dtype == 'int':
		return int(result)
	elif dtype == 'float':
		return float(result)


def get_FPGA_summary_time(fname):
	"""
	Given an FPGA summary file (summary.csv),
		return the performance number in ms for both scheduler and executor

	Key log:
	Device,Compute Unit,Kernel,Global Work Size,Local Work Size,Number Of Calls,Dataflow Execution,Max Overlapping Executions,Dataflow Acceleration,Total Time (ms),Minimum Time (ms),Average Time (ms),Maximum Time (ms),Clock Frequency (MHz),
	xilinx_u250_gen3x16_xdma_shell_4_1-0,vadd_1,vadd,1:1:1,1:1:1,1,Yes,1,1.000000x,967.647,967.647,967.647,967.647,200,
	"""
	pattern = r"(\d+.\d+)" # float
	kernel_keyword = 'xilinx_u250_gen3x16_xdma_shell_4_1-0,vadd_1,vadd,1:1:1,1:1:1,1,Yes,1,1.000000x,'
	time_ms_kernel = None
	
	with open(fname) as f:
		lines = f.readlines()
		for line in lines:
			if kernel_keyword in line:
				new_line = line.replace(kernel_keyword, "")
				time_ms_kernel = re.findall(pattern, new_line)[0]
				if '.' in time_ms_kernel:
					time_ms_kernel = float(time_ms_kernel)
				else: # only has int part, e.g., 1146,1146
					time_ms_kernel = float(re.findall(r"\d+", time_ms_kernel)[0])
	assert time_ms_kernel is not None

	return time_ms_kernel