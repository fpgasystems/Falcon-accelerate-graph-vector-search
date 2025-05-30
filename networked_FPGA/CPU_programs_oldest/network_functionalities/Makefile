CC = g++
CLAGS=-Wall
LINK = -lpthread

all: network_send \
	network_send_slow \
	network_send_per_query \
	network_recv query_send_thread \
	result_recv_thread 

network_send: network_send.cpp
	${CC} ${CLAGS} network_send.cpp ${LINK} -o network_send

network_send_slow: network_send_slow.cpp
	${CC} ${CLAGS} network_send_slow.cpp ${LINK} -o network_send_slow

network_send_per_query: network_send_per_query.cpp
	${CC} ${CLAGS} network_send_per_query.cpp ${LINK} -o network_send_per_query	

network_recv: network_recv.cpp
	${CC} ${CLAGS} network_recv.cpp ${LINK} -o network_recv

query_send_thread: query_send_thread.cpp
	${CC} ${CLAGS} query_send_thread.cpp ${LINK} -o query_send_thread

result_recv_thread: result_recv_thread.cpp
	${CC} ${CLAGS} result_recv_thread.cpp ${LINK} -o result_recv_thread


.PHONY: clean, cleanall

cleanall: clean

clean:
	rm network_send network_send_slow network_send_per_query network_recv  query_send_thread result_recv_thread 