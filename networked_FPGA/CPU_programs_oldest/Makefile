CC = g++
CLAGS=-Wall
LINK = -lpthread

all: FPGA_simulator \
	host_single_FPGA \
	host_multi_FPGA \
	hnswlib_recall \
	hnswlib_save_load_index


# network_send: network_send.cpp
# 	${CC} ${CLAGS} network_send.cpp ${LINK} -o network_send

# network_send_slow: network_send_slow.cpp
# 	${CC} ${CLAGS} network_send_slow.cpp ${LINK} -o network_send_slow

# network_send_per_query: network_send_per_query.cpp
# 	${CC} ${CLAGS} network_send_per_query.cpp ${LINK} -o network_send_per_query	

# network_recv: network_recv.cpp
# 	${CC} ${CLAGS} network_recv.cpp ${LINK} -o network_recv

# query_send_thread: query_send_thread.cpp
# 	${CC} ${CLAGS} query_send_thread.cpp ${LINK} -o query_send_thread

# result_recv_thread: result_recv_thread.cpp
# 	${CC} ${CLAGS} result_recv_thread.cpp ${LINK} -o result_recv_thread

FPGA_simulator: FPGA_simulator.cpp
	${CC} ${CLAGS} FPGA_simulator.cpp ${LINK} -o FPGA_simulator

host_single_FPGA: host_single_FPGA.cpp
	${CC} ${CLAGS} host_single_FPGA.cpp ${LINK} -o host_single_FPGA

# host_single_FPGA_SIFT: host_single_FPGA_SIFT.cpp
# 	${CC} ${CLAGS} host_single_FPGA_SIFT.cpp ${LINK} -o host_single_FPGA_SIFT

host_multi_FPGA: host_multi_FPGA.cpp
	${CC} ${CLAGS} host_multi_FPGA.cpp ${LINK} -o host_multi_FPGA

# host_single_FPGA_same_conn: host_single_FPGA_same_conn.cpp
# 	${CC} ${CLAGS} host_single_FPGA_same_conn.cpp ${LINK} -o host_single_FPGA_same_conn

# host_single_FPGA_same_conn_non_blocking: host_single_FPGA_same_conn_non_blocking.cpp
# 	${CC} ${CLAGS} host_single_FPGA_same_conn_non_blocking.cpp ${LINK} -o host_single_FPGA_same_conn_non_blocking

hnswlib_recall: hnswlib_recall.cpp
	${CC} ${CLAGS} hnswlib_recall.cpp ${LINK} -o hnswlib_recall

hnswlib_save_load_index: hnswlib_save_load_index.cpp
	${CC} ${CLAGS} hnswlib_save_load_index.cpp ${LINK} -o hnswlib_save_load_index

.PHONY: clean, cleanall

cleanall: clean

clean:
# rm network_send network_send_slow network_send_per_query network_recv  query_send_thread result_recv_thread 
	rm FPGA_simulator host_single_FPGA host_multi_FPGA hnswlib_recall hnswlib_save_load_index
# rm host_single_FPGA_same_conn host_single_FPGA_same_conn_non_blocking
