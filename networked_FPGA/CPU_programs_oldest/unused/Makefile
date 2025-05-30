CC = g++
CLAGS=-Wall
LINK = -lpthread -I../

all: host_single_FPGA_same_conn \
	host_single_FPGA_same_conn_non_blocking \


host_single_FPGA_same_conn: host_single_FPGA_same_conn.cpp
	${CC} ${CLAGS} host_single_FPGA_same_conn.cpp ${LINK} -o host_single_FPGA_same_conn

host_single_FPGA_same_conn_non_blocking: host_single_FPGA_same_conn_non_blocking.cpp
	${CC} ${CLAGS} host_single_FPGA_same_conn_non_blocking.cpp ${LINK} -o host_single_FPGA_same_conn_non_blocking


.PHONY: clean, cleanall

cleanall: clean

clean:
	rm host_single_FPGA_same_conn host_single_FPGA_same_conn_non_blocking
