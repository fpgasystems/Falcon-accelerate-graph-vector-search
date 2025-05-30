# hacc build server default gcc version is 9.4.0, does not support semaphore in c++20
# In case the following error occurs, /usr/bin/ld: final link failed: Input/output error, manually type in the g++ command
CC = /home/wejiang/.local/gcc-12.3.0/bin/g++
# CC = g++

CLAGS=-Wall -std=c++20
LINK = -lpthread
LINK_OMP = -fopenmp

all: FPGA_simulator \
	CPU_cooridnator_for_GPU_FPGA \
	CPU_index_server \
	# CPU_to_single_FPGA \
	# host_multi_FPGA \
	# hnswlib_recall \
	# hnswlib_save_load_index

CPU_cooridnator_for_GPU_FPGA: CPU_cooridnator_for_GPU_FPGA.cpp
	${CC} ${CLAGS} CPU_cooridnator_for_GPU_FPGA.cpp ${LINK} -o CPU_cooridnator_for_GPU_FPGA

FPGA_simulator: FPGA_simulator.cpp
	${CC} ${CLAGS} FPGA_simulator.cpp ${LINK} -o FPGA_simulator

CPU_index_server: CPU_index_server.cpp
	${CC} ${CLAGS} CPU_index_server.cpp ${LINK} ${LINK_OMP} -o CPU_index_server

.PHONY: clean, cleanall

cleanall: clean

clean:
	rm FPGA_simulator CPU_index_server CPU_cooridnator_for_GPU_FPGA
