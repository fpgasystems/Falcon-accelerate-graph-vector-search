

import numpy as np
import random
import time
import bisect
import argparse
import os
import glob

class Neighbor:
    def __init__(self, id, distance, flag):
        self.id = id
        self.distance = distance
        self.flag = flag

    def __lt__(self, other: 'Neighbor'):
        return self.distance < other.distance
    
    def printNeighbor(self):
        print("id: ", self.id, "distance: ", self.distance, "flag: ", self.flag)


def compare(query_data, db_vec, dimension: int):
    return np.sum((query_data - db_vec) ** 2)


def print_retset(retset):
    for i in range(len(retset)):
        retset[i].printNeighbor()
        
        
class IndexNSG():
    def __init__(self, dimension: int, num_nodes: int):
        self.final_graph = []
        self.dimension = dimension
        self.width = 0 # M in HNSW, max edges
        self.ep_ = 0
        self.cur_element_count = num_nodes
    
    def Load(self, filename: str):
        with open(filename, "rb") as f:
            self.width = int.from_bytes(f.read(4), byteorder='little', signed=False)
            self.ep_ = int.from_bytes(f.read(4), byteorder='little', signed=False)
            cc = 0
            while True:
                k_bytes = f.read(4)
                if not k_bytes:
                    break
                k = int.from_bytes(k_bytes, byteorder='little', signed=False)
                cc += k
                tmp = list(np.frombuffer(f.read(k * 4), dtype=np.uint32))
                self.final_graph.append(tmp)
            cc //= self.cur_element_count
            # print(cc)
        assert self.cur_element_count == len(self.final_graph)
        
        print("NSG index file loaded")
        print("width: ", self.width)
        print("ep_: ", self.ep_)

    def search_with_base_graph(self, query, x, K, parameters):
        L = parameters['L_search']
        data_ = x
        retset:list[Neighbor] = []
        init_ids = []
        flags = [0] * self.cur_element_count
        node_counter = 0
        
        for tmp_l in range(min(L, len(self.final_graph[self.ep_]))):
            init_ids.append(self.final_graph[self.ep_][tmp_l])
            flags[init_ids[tmp_l]] = 1
        
        tmp_l += 1
    
        while tmp_l < L:
            id = random.randint(0, self.cur_element_count - 1)
            if flags[id] == 1:
                continue
            flags[id] = 1
            init_ids.append(id)
            tmp_l += 1

        for i in range(len(init_ids)):
            id = init_ids[i]
            dist = compare(data_[id], query, self.dimension)
            retset.append(Neighbor(id, dist, 1))
        
        node_counter += len(init_ids)
        
        retset.sort()
        
        k = 0
        while k < L:
            nk = L
            if retset[k].flag:
                retset[k].flag = 0
                n = retset[k].id
        
                for m in range(len(self.final_graph[n])):
                    id = self.final_graph[n][m]
                    if flags[id] == 1:
                        continue
                    flags[id] = 1
                    dist = compare(query, data_[id], self.dimension)
                    node_counter += 1
                    if dist >= retset[L - 1].distance:
                        continue
                    nn = Neighbor(id, dist, 1)
                    bisect.insort_left(retset, nn)
                    r = retset.index(nn)
                    if len(retset) > L + 1:
                        retset.pop()
                    if r < nk:
                        nk = r
                    # nk: the index of the smallest dist
            if nk <= k:
                k = nk
            else:  
                k += 1
        indices = [0] * K
        for i in range(K):
            indices[i] = retset[i].id
        
        return (indices, node_counter)


    def search_with_base_graph_2queue(self, query, x, K, parameters):
        L = parameters['L_search']
        data_ = x
        candidate_set:list[Neighbor] = []
        top_candidates:list[Neighbor] = []
        node_counter = 0
        
        init_ids = []
        flags = [0] * self.cur_element_count
        
        for tmp_l in range(min(L, len(self.final_graph[self.ep_]))):
            init_ids.append(self.final_graph[self.ep_][tmp_l])
            flags[init_ids[tmp_l]] = 1
    
        while tmp_l < L:
            id = random.randint(0, self.cur_element_count - 1)
            if flags[id] == 1:
                continue
            flags[id] = 1
            init_ids.append(id)
            tmp_l += 1

        for i in range(len(init_ids)):
            id = init_ids[i]
            dist = compare(data_[id], query, self.dimension)
            candidate_set.append(Neighbor(id, dist, 1))
            top_candidates.append(Neighbor(id, dist, 1))
        
        node_counter += len(init_ids)
        
        candidate_set.sort()
        top_candidates.sort()
        
        while len(candidate_set) > 0:
            cur_node = candidate_set.pop(0)
            
            if cur_node.distance > top_candidates[L - 1].distance:  # here we assume both queues has infinite capacity
                break
            
            cur_id = cur_node.id

            for m in range(len(self.final_graph[cur_id])):
                id = self.final_graph[cur_id][m]
                if flags[id] == 1:
                    continue
                flags[id] = 1
                dist = compare(query, data_[id], self.dimension)
                node_counter += 1
                if dist >= top_candidates[L - 1].distance:  # here L-1 is a relaxed condition ???
                    continue
                nn = Neighbor(id, dist, 1)
                bisect.insort_left(candidate_set, nn)
                bisect.insort_left(top_candidates, nn)
        
        indices = [0] * K
        for i in range(K):
            indices[i] = top_candidates[i].id
        
        return (indices, node_counter)