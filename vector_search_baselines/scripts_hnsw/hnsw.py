import hnswlib
import numpy as np 
import struct
import heapq
import time
import sys
import os

from pathlib import Path

def convertBytes(bytestring, dtype='int'):
    """
    convert bytes to a single element
    dtype = {int, long, float, double}
    struct: https://docs.python.org/3/library/struct.html
    """ 
    # int from bytes is much faster than struct.unpack
    if dtype =='int' or dtype == 'long': 
        return int.from_bytes(bytestring, byteorder='little', signed=False)
    elif dtype == 'float': 
        return struct.unpack('f', bytestring)[0]
    elif dtype == 'double': 
        return struct.unpack('d', bytestring)[0]
    else:
        raise ValueError 

# Wenqi: the fastest way to load a bytestring list is to use *** np.frombuffer ***
def convertBytesList(bytestring, dtype='int'):
    """
    Given a byte string, return the value list
    """
    result_list = []
    if dtype == 'int' or dtype == 'float':
        dsize = 4
    elif dtype == 'long' or dtype == 'double':
        dsize = 8
    else:
        raise ValueError 
        
    start_pointer = 0
    for i in range(len(bytestring) // dsize):
        result_list.append(convertBytes(
            bytestring[start_pointer: start_pointer + dsize], dtype=dtype))
        start_pointer += dsize
    return result_list

def calculateDist(query_data, db_vec):
    """
    HNSWLib returns L2 square distance, so do we
        both inputs are 1-d np array
    """
    # return l2 distance between two points
    return np.sum((query_data - db_vec) ** 2)


def merge_two_distance_list(list_A, list_B, k):
    """
    merge two lists by selecting the k pairs of the smallest distance
    input:
        both list has format [(dist, ID), (dist, ID), ...]
    return:
        a result list, with ascending distance (the first contains the largest distance)
    """
    
    results_heap = []
    for i in range(len(list_A)):
        dist, server_ID, vec_ID = list_A[i]
        heapq.heappush(results_heap, (-dist, server_ID, vec_ID))
    for i in range(len(list_B)):
        dist, server_ID, vec_ID = list_B[i]
        heapq.heappush(results_heap, (-dist, server_ID, vec_ID))

    while len(results_heap) > k:
        heapq.heappop(results_heap)

    results = []
    while len(results_heap) > 0:
        dist, server_ID, vec_ID = results_heap[0]
        results.append((-dist, server_ID, vec_ID))
        heapq.heappop(results_heap)
    results.reverse()
            
    return results

 
class HNSW_index():
    
    """
    Returned result list always in the format of (dist, server_ID, vec_ID),
        in ascending distance order (the first result is the nearest neighbor)
    """
    
    def __init__(self, local_server_ID=0, dim=128):
        
        self.dim = dim
        self.local_server_ID = local_server_ID
        
        # Meta Info
        self.offsetLevel0_ = None
        self.max_elements_ = None
        self.cur_element_count = None
        self.size_data_per_element_ = None
        self.label_offset_ = None
        self.offsetData_ = None
        self.maxlevel_ = None
        self.enterpoint_node_ = None
        self.maxM_ = None
        self.maxM0_ = None
        self.M_ = None
        self.mult_ = None # the probability that a node is one a higher level
        self.ef_construction_ = None
        
        # ground layer, all with length of cur_element_count
        self.links_count_l0 = None # a list of link_count
        self.links_l0 = None # a list of links per vector
        self.data_l0 = None # a list of vectors
        label_l0 = None # a list of vector IDs
        
        # upper layers, all with length of cur_element_count
        self.element_levels_ = None # the level per vector
        self.links = None # the upper layer link info (link count + links)
        
        # remote nodes, order according to local ID (not label ID)
        #  remote_links: an 2-D array (cur_element_count, k), 
        #    each element is a tuple: (server_ID, vector_ID)
        self.remote_links_count = None
        self.remote_links = None

    def load_index_and_data(self, index_path):
        """
        Load index & data from the saved HNSW index binary file
        """
        index = Path(index_path).read_bytes()
        self.load_meta_info(index)
        self.load_ground_layer(index)
        self.load_upper_layers(index)
        print("Index loaded")

    def load_meta_info(self, index_bin):
        """
        index_bin = hnswlib index binary 
        
        HNSW save index order:
            https://github.com/WenqiJiang/hnswlib-eval/blob/master/hnswlib/hnswalg.h#L588-L616
        """
        self.offsetLevel0_ = int.from_bytes(index_bin[0:8], byteorder='little', signed=False)
        self.max_elements_ = int.from_bytes(index_bin[8:16], byteorder='little', signed=False)
        self.cur_element_count = int.from_bytes(index_bin[16:24], byteorder='little', signed=False)
        self.size_data_per_element_ = int.from_bytes(index_bin[24:32], byteorder='little', signed=False)
        self.label_offset_ = int.from_bytes(index_bin[32:40], byteorder='little', signed=False)
        self.offsetData_ = int.from_bytes(index_bin[40:48], byteorder='little', signed=False)
        self.maxlevel_ = int.from_bytes(index_bin[48:52], byteorder='little', signed=False)
        self.enterpoint_node_ = int.from_bytes(index_bin[52:56], byteorder='little', signed=False)
        self.maxM_ = int.from_bytes(index_bin[56:64], byteorder='little', signed=False)
        self.maxM0_ = int.from_bytes(index_bin[64:72], byteorder='little', signed=False)
        self.M_ = int.from_bytes(index_bin[72:80], byteorder='little', signed=False)
        self.mult_ = struct.unpack('d', index_bin[80:88])[0] # the probability that a node is one a higher level
        self.ef_construction_ = int.from_bytes(index_bin[88:96], byteorder='little', signed=False)
        

        print("offsetLevel0_", self.offsetLevel0_)
        print("max_elements_", self.max_elements_)
        print("cur_element_count", self.cur_element_count)
        print("size_data_per_element_", self.size_data_per_element_)
        print("label_offset_", self.label_offset_)
        print("offsetData_", self.offsetData_)
        print("maxlevel_", self.maxlevel_)
        print("enterpoint_node_", self.enterpoint_node_)
        print("maxM_", self.maxM_)
        print("maxM0_", self.maxM0_)
        print("M_", self.M_)
        print("mult_", self.mult_)
        print("ef_construction_", self.ef_construction_)
        
    
    def load_ground_layer(self, index_bin):
        """
        Get the ground layer vector ID, vectors, and links:
            links_count_l0: vec_num
            links_l0: maxM0_ * vec_num 
            data_l0: (dim, vec_num)
            label_l0: vec_num
        """
        
        # Layer 0 data 
        start_byte_pointer = 96
        delta = self.cur_element_count * self.size_data_per_element_
        data_level0 = index_bin[start_byte_pointer: start_byte_pointer + delta]
        
        size = len(data_level0)
        self.links_count_l0 = []
        self.links_l0 = np.zeros((self.cur_element_count, self.maxM0_), dtype=int)
        self.data_l0 = np.zeros((self.cur_element_count, self.dim))
        self.label_l0 = []

        data_l0_list = []
        
        assert len(data_level0) == self.size_data_per_element_ * self.cur_element_count
        
        size_link_count = 4
        size_links = self.maxM0_ * 4
        size_vectors = self.dim * 4
        size_label = 8
        
        assert self.size_data_per_element_ == \
            size_link_count + size_links + size_vectors + size_label
            
        for i in range(self.cur_element_count):
            # per ground layer node: (link_count (int), links (int array of len=maxM0_), 
            #    vector (float array of len=dim, vector ID (long)))
            
            addr_link_count = i * self.size_data_per_element_ 
            addr_links = addr_link_count + size_link_count
            addr_vectors = addr_links + size_links
            addr_label = addr_vectors + size_vectors
            
            tmp_bytes = data_level0[addr_link_count: addr_link_count + size_link_count]
            self.links_count_l0.append(convertBytes(tmp_bytes, dtype='int'))
        
            tmp_bytes = data_level0[addr_links: addr_links + size_links]
            self.links_l0[i] = np.frombuffer(tmp_bytes, dtype=np.int32)
            
            tmp_bytes = data_level0[addr_vectors: addr_vectors + size_vectors]
            self.data_l0[i] = np.frombuffer(tmp_bytes, dtype=np.float32)
            
            tmp_bytes = data_level0[addr_label: addr_label + size_label]
            self.label_l0.append(convertBytes(tmp_bytes, dtype='long'))


    def load_upper_layers(self, index_bin):
        """
        Get the upper layer info:
            element_levels_: the levels of each vector
            links: list of upper links
        """
        
        # meta + ground data
        start_byte_pointer = 96 + self.max_elements_ * self.size_data_per_element_
        
        # Upper layers
        size_links_per_element_ = self.maxM_ * 4 + 4
        self.element_levels_ = []
        self.links = []

        for i in range(self.cur_element_count):
            tmp_bytes = index_bin[start_byte_pointer:start_byte_pointer+4]
            linkListSize = convertBytes(tmp_bytes, dtype='int')
            start_byte_pointer += 4
            
            # if an element is only on ground layer, it has no links on upper layers at all
            if linkListSize == 0:
                self.element_levels_.append(0)
                self.links.append([])
            else:
                level = int(linkListSize / size_links_per_element_)
                self.element_levels_.append(level)
                tmp_bytes = index_bin[start_byte_pointer:start_byte_pointer+linkListSize]
                links_tmp = list(np.frombuffer(tmp_bytes, dtype=np.int32))
                start_byte_pointer += linkListSize
                self.links.append(links_tmp)

        assert start_byte_pointer == len(index_bin) # 6606296

    def save_as_FPGA_format(self, out_dir, num_channels=[1]):
        """
        Save the data as FPGA format, must make sure `load_ground_layer` and `load_upper_layers` are already invoked.
        """
        print("Converting HNSW index to FPGA format")
        # save meta data
        byte_array_meta = bytearray() 
        byte_array_meta += self.cur_element_count.to_bytes(4, byteorder='little', signed=False)
        byte_array_meta += self.maxlevel_.to_bytes(4, byteorder='little', signed=False)
        byte_array_meta += self.enterpoint_node_.to_bytes(4, byteorder='little', signed=False)
        byte_array_meta += self.maxM_.to_bytes(4, byteorder='little', signed=False)
        byte_array_meta += self.maxM0_.to_bytes(4, byteorder='little', signed=False)

        # save ground level links
        # format of each node:
        #   [64 B header = num_links (4B int) + 60 byte paddings] + N [64B actual links] + paddings (to 64 B)
        if self.maxM0_ % 16 == 0:
            num_bytes_per_ground_links = int(self.maxM0_ / 16) * 64 + 64
        else:
            num_bytes_per_ground_links = (int(self.maxM0_ / 16) + 1) * 64 + 64
        byte_array_ground_links = bytearray()
        for i in range(self.cur_element_count):
            link_count = self.links_count_l0[i]
            links = self.links_l0[i]
            byte_array_ground_links += int(link_count).to_bytes(4, byteorder='little', signed=False)
            byte_array_ground_links += b'\x00' * 60
            for j in range(self.maxM0_):
                if j < link_count:
                    byte_array_ground_links += int(links[j]).to_bytes(4, byteorder='little', signed=False)
                else:
                    byte_array_ground_links += b'\x00' * 4
            if len(byte_array_ground_links) % 64 != 0:
                byte_array_ground_links += b'\x00' * (64 - len(byte_array_ground_links) % 64)
        assert len(byte_array_ground_links) == self.cur_element_count * num_bytes_per_ground_links
            
        # with open(os.path.join(out_dir, 'ground_links.bin'), 'wb') as f:
        #     f.write(byte_array_ground_links)

        # save ground level vectors
        # format of each node:
        # [vector (4B float) + padding] + [visited (4B int, init as -1) + padding]
        if self.dim % 16 == 0:
            num_bytes_per_vector_with_padding = int(self.dim / 16) * 64 + 64
        else:
            num_bytes_per_vector_with_padding = (int(self.dim / 16) + 1) * 64 + 64
        byte_array_ground_vectors = bytearray()
        for i in range(self.cur_element_count):
            vector = self.data_l0[i]
            for j in range(self.dim):
                byte_array_ground_vectors += struct.pack('f', vector[j])
            if len(byte_array_ground_vectors) % 64 != 0:
                byte_array_ground_vectors += b'\x00' * (64 - len(byte_array_ground_vectors) % 64)
			# -1 used for visited flag
            byte_array_ground_vectors += b'\xff\xff\xff\xff' # -1 in int32
            byte_array_ground_vectors += b'\x00' * 60
        assert len(byte_array_ground_vectors) == self.cur_element_count * num_bytes_per_vector_with_padding

        # with open(os.path.join(out_dir, 'ground_vectors.bin'), 'wb') as f:
        #     f.write(byte_array_ground_vectors)
        
        # partition to N channels in round robin fashion
        for nc in num_channels:
            byte_array_ground_vectors_per_channel = [bytearray() for _ in range(nc)]
            byte_array_ground_links_per_channel = [bytearray() for _ in range(nc)]
            for i in range(self.cur_element_count):
                byte_array_ground_vectors_per_channel[i % nc] += byte_array_ground_vectors[i * num_bytes_per_vector_with_padding: (i + 1) * num_bytes_per_vector_with_padding]
                byte_array_ground_links_per_channel[i % nc] += byte_array_ground_links[i * num_bytes_per_ground_links: (i + 1) * num_bytes_per_ground_links]
            for i in range(nc):
                with open(os.path.join(out_dir, 'ground_vectors_{}_chan_{}.bin'.format(nc, i)), 'wb') as f:
                    f.write(byte_array_ground_vectors_per_channel[i])
                with open(os.path.join(out_dir, 'ground_links_{}_chan_{}.bin'.format(nc, i)), 'wb') as f:
                    f.write(byte_array_ground_links_per_channel[i])
            
        # save ground level labels
        # each node's physical position will be translates to its label
        byte_array_ground_labels = bytearray()
        for i in range(self.cur_element_count):
            label = self.label_l0[i]
            byte_array_ground_labels += int(label).to_bytes(4, byteorder='little', signed=False)
        assert len(byte_array_ground_labels) == self.cur_element_count * 4
        
        # save upper links, format:
        #   [num_links (4B int) + padding] + N [64B actual links] + paddings (to 64 B)
        byte_array_upper_links = bytearray()
        # save pointers to addresses of upper links, each is 4B int as a pointer to a byte address
        byte_array_upper_links_pointers = bytearray()
        for i in range(self.cur_element_count):
            levels = self.element_levels_[i] 
            pointer_addr = len(byte_array_upper_links) 
            byte_array_upper_links_pointers += pointer_addr.to_bytes(8, byteorder='little', signed=False)
            if levels != 0:
                links = self.links[i]
                for j in range(levels):
                    link_count = links[j * (1 + self.M_)]
                    byte_array_upper_links += int(link_count).to_bytes(4, byteorder='little', signed=False)
                    byte_array_upper_links += b'\x00' * 60
                    for k in range(self.M_):
                        byte_array_upper_links += int(links[j * (1 + self.M_) + 1 + k]).to_bytes(4, byteorder='little', signed=False)
                    if len(byte_array_upper_links) % 64 != 0:
                        byte_array_upper_links += b'\x00' * (64 - len(byte_array_upper_links) % 64)
        
        # save to files
        with open(os.path.join(out_dir, 'meta.bin'), 'wb') as f:
            f.write(byte_array_meta)
        with open(os.path.join(out_dir, 'ground_labels.bin'), 'wb') as f:
            f.write(byte_array_ground_labels)
        with open(os.path.join(out_dir, 'upper_links.bin'), 'wb') as f:
            f.write(byte_array_upper_links)
        with open(os.path.join(out_dir, 'upper_links_pointers.bin'), 'wb') as f:
            f.write(byte_array_upper_links_pointers)

        print("FPGA format saved")
        
        return byte_array_meta, byte_array_ground_links, byte_array_ground_labels, byte_array_ground_vectors, byte_array_upper_links, byte_array_upper_links_pointers

    def searchKnn(self, q_data, k, ef, debug=False):
        """
        result a list of (distance, vec_ID) in ascending distance
        """
        
        ep_node = self.enterpoint_node_
        num_elements = self.cur_element_count
        max_level = self.maxlevel_
        links_count_l0 = self.links_count_l0
        links_l0 = self.links_l0
        data_l0 = self.data_l0
        links = self.links
        label_l0 = self.label_l0
        dim = self.dim
        
        currObj = ep_node
        currVec = data_l0[currObj]
        curdist = calculateDist(q_data, currVec)
        
        search_path_local_ID = set()
        search_path_vec_ID = set()
        
        # search upper layers
        for level in reversed(range(1, max_level+1)):
            # if debug:
            #     print("")
            #     print("level: ", level)
            changed = True
            while changed:
                # if debug:
                #     print("current object: ", currObj, ", current distance: ", curdist)
                search_path_local_ID.add(currObj)
                changed = False
                ### Wenqi: here, assuming Node ID can be used to retrieve upper links (which is not true for indexes with ID starting from non-0)
                if (len(links[currObj])==0):
                    break
                else:
                    start_index = (level-1) * (1 + self.M_)
                    size = links[currObj][start_index]
                    # if debug:
                    #     print("size of neighbors: ", size) 
                    neighbors = links[currObj][(start_index+1):(start_index+(1 + self.M_))]
                    for i in range(size):
                        cand = neighbors[i]
                        currVec = data_l0[cand]
                        dist = calculateDist(q_data, currVec)
                        # if debug:
                        #     print("cand: ", cand, ", dist: ", dist)
                        if (dist < curdist):
                            curdist = dist
                            currObj = cand
                            changed = True
                    #         if debug:
                    #             print("changed")
                    # if debug:
                    #     print("one node finish")
                    #     print("")

        # search in ground layer
        if debug:
            print("level 0 entry point: ", currObj)
            print("upper level hops: ", len(search_path_local_ID))
        visited_array = set() # default 0
        top_candidates = []
        candidate_set = []
        lowerBound = curdist 
        # By default heap queue is a min heap: https://docs.python.org/3/library/heapq.html
        # candidate_set = candidate list, min heap
        # top_candidates = dynamic list (potential results), max heap
        # compare min(candidate_set) vs max(top_candidates)
        heapq.heappush(top_candidates, (-curdist, currObj))
        heapq.heappush(candidate_set,(curdist, currObj))
        visited_array.add(currObj) 

        cnt_hops = 0
        max_cand_size = 0
        
        while len(candidate_set)!=0:
            current_node_pair = candidate_set[0]
            if ((current_node_pair[0] > lowerBound)):
                break
            else:
                cnt_hops += 1
                max_cand_size = max(max_cand_size, len(candidate_set))
            heapq.heappop(candidate_set)
            current_node_id = current_node_pair[1]
            search_path_local_ID.add(current_node_id)
            size = links_count_l0[current_node_id]
            num_visted_neighbors_this_iter = 0
            # if debug:
            #     print("current object: ", current_node_id)
            #     print("size of neighbors: ", size)
            for i in range(size):
                candidate_id = links_l0[current_node_id][i]
                if (candidate_id not in visited_array):
                    visited_array.add(candidate_id)
                    currVec = data_l0[candidate_id]
                    dist = calculateDist(q_data, currVec)
                    num_visted_neighbors_this_iter += 1
                    # if debug:
                    #     print("current object: ", candidate_id, ", current distance: ", dist, ", lowerBound: ", lowerBound)
                    if (len(top_candidates) < ef or lowerBound > dist):
                        # if debug:
                        #     print("added")
                        heapq.heappush(candidate_set, (dist, candidate_id))
                        heapq.heappush(top_candidates, (-dist, candidate_id))
                    if (len(top_candidates) > ef):
                        heapq.heappop(top_candidates)
                    if (len(top_candidates)!=0):
                        lowerBound = -top_candidates[0][0]
                else :
                    # if debug:
                    #     print("current object: ", candidate_id, ", visited already")
                    pass
            # if debug:
            #     if num_visted_neighbors_this_iter == 0:
            #         print("current object: ", current_node_id, ", no unvisited neighbors")
            #     print("one node finishes")
                # print("")
        if debug:
            print("total hops: ", cnt_hops)
            print("max candidate set size: ", max_cand_size)
            print("total visited nodes (filtered out seen nodes): ", len(visited_array))

        while len(top_candidates) > k:
            heapq.heappop(top_candidates)

        result = []
        while len(top_candidates) > 0:
            candidate_pair = top_candidates[0]
            # Wenqi: here, replace the local candidate ID by real node ID, great!
            result.append([-candidate_pair[0], self.local_server_ID, label_l0[candidate_pair[1]]])
            heapq.heappop(top_candidates)
        result.reverse()
            
        for local_ID in search_path_local_ID:
            search_path_vec_ID.add(label_l0[local_ID])

        return result, search_path_local_ID, search_path_vec_ID

