import numpy as np

def safe_div(n, d):
    return n / d if d else 0

class Resources:

    def __init__(self, BRAM_18K=0, DSP=0, FF=0, LUT=0, URAM=0):
        self.BRAM_18K = BRAM_18K
        self.DSP = DSP
        self.FF = FF
        self.LUT = LUT
        self.URAM = URAM
    
    def add_resources(resource_list):
        """
        Return the sum of two resources.
        """
        sum_resources = Resources()
        for resource in resource_list:
            sum_resources.BRAM_18K += resource.BRAM_18K
            sum_resources.DSP += resource.DSP
            sum_resources.FF += resource.FF
            sum_resources.LUT += resource.LUT
            sum_resources.URAM += resource.URAM

        return sum_resources

    def __str__(self):
        return f"BRAM_18K: {self.BRAM_18K}, DSP: {self.DSP}, FF: {self.FF}, LUT: {self.LUT}, URAM: {self.URAM}"

    def calculate_percentage(self, resource_total):
        """
        Calculate the percentage of the current object compared to the total resources.
        """
        percentage = Resources()
        percentage.BRAM_18K = safe_div(self.BRAM_18K, resource_total.BRAM_18K)
        percentage.DSP = safe_div(self.DSP, resource_total.DSP)
        percentage.FF = safe_div(self.FF, resource_total.FF)
        percentage.LUT = safe_div(self.LUT, resource_total.LUT)
        percentage.URAM = safe_div(self.URAM, resource_total.URAM)

        # print(percentage)

        return percentage

# Controller
resource_task_scheduler = Resources(BRAM_18K=1, DSP=0, FF=38670, LUT=40116, URAM=0)
resource_results_collection = Resources(BRAM_18K=0, DSP=0, FF=46100, LUT=53986, URAM=0)
resource_controller = Resources.add_resources([resource_task_scheduler, resource_results_collection])

# Fetch neighbors
resource_fetch_neighbor_ids = Resources(BRAM_18K=8, DSP=3, FF=1413, LUT=6933, URAM=0)

# Bloom filter
resource_check_update = Resources(BRAM_18K=0, DSP=0, FF=600, LUT=2033, URAM=0)
n_hash = 3
resource_stream_hash = Resources(BRAM_18K=0, DSP=n_hash * 12, FF=n_hash * 383, LUT=n_hash * 696, URAM=0)
resource_bloom_filter = Resources.add_resources([resource_check_update, resource_stream_hash])

# Fetch vectors
resource_fetch_vectors = Resources(BRAM_18K=0, DSP=0, FF=945, LUT=868, URAM=0)

# Compute
resource_compute_distances = Resources(BRAM_18K=26, DSP=122, FF=29080, LUT=21517, URAM=0)

# Total resources
def get_total_resources(mode='inter-query', n_BFC=4):
    if mode == 'inter-query':
        resource_total = Resources.add_resources(
            n_BFC * [resource_controller, resource_fetch_neighbor_ids, resource_bloom_filter, resource_fetch_vectors, resource_compute_distances])
    elif mode == 'intra-query':
        resource_total = Resources.add_resources(
            [resource_controller, resource_fetch_neighbor_ids] +
            n_BFC * [resource_bloom_filter, resource_fetch_vectors, resource_compute_distances])
    
    # Calculate the percentage of resources used by each task
    print("Controller resource ratio:", resource_controller.calculate_percentage(resource_total))
    print("Bloom filter resource ratio:", resource_bloom_filter.calculate_percentage(resource_total))
    print("Compute distances resource ratio:", resource_compute_distances.calculate_percentage(resource_total))
     
    return resource_total


def calculate_utilization_control(freq, qps, avg_hops, avg_visited):
    """
    Return the utilization ratio of the control units
    """
    # number of insertions per sec = #visited nodes * qps
    # number of sortings per sec = #hops * qps
    num_insertions = avg_visited * qps
    num_sortings = avg_hops * qps
    cycles_per_insertion = 1
    cycles_per_sorting = 64
    effective_cycles = num_insertions * cycles_per_insertion + num_sortings * cycles_per_sorting
    # number of cycles per sec = freq
    utilization = safe_div(effective_cycles, freq)
    return utilization

def calculate_utilization_bloom_filter(freq, qps, avg_hops, avg_visited):
    """
    Return the utilization ratio of the bloom filter
    """
    # number of checks per sec = #visited nodes * qps
    num_checks = avg_visited * qps
    cycles_per_check = 3 * 2 # 3 hashes, read + write
    effective_cycles = num_checks * cycles_per_check
    # number of cycles per sec = freq
    utilization = safe_div(effective_cycles, freq)
    return utilization

def calculate_utilization_compute_distances(freq, qps, avg_hops, avg_visited, d):
    """
    Return the utilization ratio of the compute distances
    """
    # numbers of distance computations per sec = avg_visited * qps
    num_computations = avg_visited * qps
    cycles_per_computation = d / 16 # 16 elements per cycle
    effective_cycles = num_computations * cycles_per_computation
    # number of cycles per sec = freq
    utilization = safe_div(effective_cycles, freq)
    return utilization

# def calculate_utilization_total(freq, qps, avg_hops, avg_visited, d):
#     """
#     Return the total utilization ratio of the accelerator
#     """
#     pass

def calculate_silicon_efficiency(freq, qps, avg_hops, avg_visited, d, mode='inter-query', n_BFC=4):
    """
    Calculate the silicon efficiency of the accelerator.
    """
    assert mode in ['inter-query', 'intra-query']
    util_control = calculate_utilization_control(freq, qps, avg_hops, avg_visited)
    if mode == 'inter-query':
        util_control /= n_BFC
    util_bloom_filter = calculate_utilization_bloom_filter(freq, qps, avg_hops, avg_visited) / n_BFC
    util_compute_distances = calculate_utilization_compute_distances(freq, qps, avg_hops, avg_visited, d) / n_BFC
    print("Control utilization: {:.2f}%".format(util_control * 100))
    print("Bloom filter utilization: {:.2f}%".format(util_bloom_filter * 100))
    print("Compute distances utilization: {:.2f}%".format(util_compute_distances * 100))

    # # calculated weighted resource consumption
    # resource_controller
    # # Fetch neighbors
    # resource_fetch_neighbor_ids 
    # # Bloom filter
    # resource_bloom_filter
    # # Fetch vectors
    # resource_fetch_vectors 
    # # Compute
    # resource_compute_distances 
    weighted_util = 0
    if mode == 'inter-query':
        resource_total = get_total_resources(mode='inter-query', n_BFC=n_BFC)
        weighted_util += resource_controller.calculate_percentage(resource_total).LUT * util_control * n_BFC
        weighted_util += resource_bloom_filter.calculate_percentage(resource_total).LUT * util_bloom_filter * n_BFC
        weighted_util += resource_compute_distances.calculate_percentage(resource_total).LUT * util_compute_distances * n_BFC
    elif mode == 'intra-query':
        resource_total = get_total_resources(mode='intra-query', n_BFC=n_BFC)
        weighted_util += resource_controller.calculate_percentage(resource_total).LUT * util_control
        weighted_util += resource_bloom_filter.calculate_percentage(resource_total).LUT * util_bloom_filter * n_BFC
        weighted_util += resource_compute_distances.calculate_percentage(resource_total).LUT * util_compute_distances * n_BFC
    print("Weighted utilization: {:.2f}%".format(weighted_util * 100))

    





"""
    graph_type  dataset max_degree  ef max_cand_per_group max_group_num_in_pipe  time_ms_kernel  recall_1  recall_10  avg_hops  avg_visited
0         HNSW   SIFT1M         64  64                  1                     1         641.453    0.9939    0.98012   70.0091     1745.550
1         HNSW   SIFT1M         64  64                  1                     2         432.920    0.9937    0.98079   76.1094     1897.700
2         HNSW   SIFT1M         64  64                  1                     3         359.513    0.9941    0.98132   81.9769     2041.470
3         HNSW   SIFT1M         64  64                  1                     4         336.277    0.9945    0.98176   87.7296     2181.500
4         HNSW   SIFT1M         64  64                  1                     5         338.155    0.9943    0.98223   93.4252     2317.520
5         HNSW   SIFT1M         64  64                  1                     6         349.488    0.9944    0.98283   99.0498     2450.980
6         HNSW   SIFT1M         64  64                  2                     1         487.520    0.9939    0.98069   74.9309     1873.300
7         HNSW   SIFT1M         64  64                  2                     2         360.977    0.9944    0.98177   86.6301     2158.350
8         HNSW   SIFT1M         64  64                  2                     3         351.339    0.9943    0.98281   98.0128     2429.460
9         HNSW   SIFT1M         64  64                  2                     4         377.955    0.9945    0.98361  109.1660     2691.210
10        HNSW   SIFT1M         64  64                  2                     5         409.247    0.9947    0.98416  120.1730     2946.890
11        HNSW   SIFT1M         64  64                  2                     6         440.614    0.9945    0.98471  131.1770     3198.620
12        HNSW   SIFT1M         64  64                  3                     1         440.928    0.9940    0.98098   79.9239     1999.420
13        HNSW   SIFT1M         64  64                  3                     2         359.932    0.9944    0.98269   97.2421     2414.430
14        HNSW   SIFT1M         64  64                  3                     3         393.020    0.9948    0.98377  114.0220     2806.250
15        HNSW   SIFT1M         64  64                  3                     4         439.819    0.9948    0.98465  130.6050     3186.820
16        HNSW   SIFT1M         64  64                  3                     5         484.517    0.9950    0.98564  146.2620     3544.480
17        HNSW   SIFT1M         64  64                  3                     6         527.812    0.9941    0.98643  160.5610     3884.140
18        HNSW   SIFT1M         64  64                  4                     1         425.007    0.9940    0.98134   85.0507     2127.120
19        HNSW   SIFT1M         64  64                  4                     2         381.823    0.9946    0.98358  107.9130     2666.660
20        HNSW   SIFT1M         64  64                  4                     3         439.660    0.9946    0.98476  130.1560     3177.680
21        HNSW   SIFT1M         64  64                  4                     4         498.044    0.9949    0.98585  150.4310     3646.010
22        HNSW   SIFT1M         64  64                  4                     5         555.324    0.9941    0.98647  169.5420     4096.260
23        HNSW   SIFT1M         64  64                  4                     6         613.251    0.9936    0.98695  189.1120     4544.690
baseline (mc = 1, mg = 1): time_ms_kernel: 641.453, recall_1: 0.9939, recall_10: 0.98012, avg_hops: 70.0091, avg_visited: 1745.550
best dst  (mc = 1, mg = 4): time_ms_kernel: 336.277, recall_1: 0.9945, recall_10: 0.98176, avg_hops: 87.7296, avg_visited: 2181.500
"""

# Calculate the silicon efficiency of the accelerator
freq = 200 * 10**6
nq = 10000
n_BFC = 4
d = 128

print("\n==== Silicon efficiency ====")

print("\nBaseline (inter-query):")
time_ms_kernel = 641.453
qps = nq * 1000 / time_ms_kernel
avg_hops = 70.0091
avg_visited = 1745.550
calculate_silicon_efficiency(freq, qps, avg_hops, avg_visited, d, mode='inter-query', n_BFC=n_BFC)

# Inter query:
# 75        HNSW   SIFT1M         64  64                  1                     4         336.261    0.9945    0.98176   87.7296     2181.500
print("\nDST (inter-query):")
time_ms_kernel = 336.277
qps = nq * 1000 / time_ms_kernel 
avg_hops = 87.7296
avg_visited = 2181.500
calculate_silicon_efficiency(freq, qps, avg_hops, avg_visited, d, mode='inter-query', n_BFC=n_BFC)

# Intra query:
# 0         HNSW   SIFT1M         64  64                  1                     1        1814.540    0.9938    0.98012   70.0090        None
print("\nBaseline (intra-query):")
time_ms_kernel = 1814.540
qps = nq * 1000 / time_ms_kernel
avg_hops = 70.0091
avg_visited = 1745.550
calculate_silicon_efficiency(freq, qps, avg_hops, avg_visited, d, mode='intra-query', n_BFC=n_BFC)

# Inter query:
# 85        HNSW   SIFT1M         64  64                  2                     6         440.623    0.9945    0.98471  131.1770     3198.620
# Intra query:
# 13        HNSW   SIFT1M         64  64                  2                     6         674.088    0.9953    0.98543  131.1840        None
print("\nDST (intra-query):")
time_ms_kernel = 674.088
qps = nq * 1000 / time_ms_kernel
avg_hops = 131.1770
avg_visited = 3198.620
calculate_silicon_efficiency(freq, qps, avg_hops, avg_visited, d, mode='intra-query', n_BFC=n_BFC)
