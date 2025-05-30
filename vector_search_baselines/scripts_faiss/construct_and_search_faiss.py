"""
Benchmarking the CPU's throughput using 10,000 queries (takes several minutes),

There are 2 ways to use the script:

(1) Mode A: test the performance of nprobe & topK & qbs:

python construct_and_search_faiss.py --faiss_index_path ../data/CPU_Faiss_indexes --dbname SIFT1M --index_key IVF1024,Flat --mode run_one --topK 100 --qbs 100 --use_gpu 0 --parametersets 'nprobe=1 nprobe=32 nprobe=64 nprobe=128'

optional: --nthreads 1

(2) Mode B: test the throughput of a set of nprobe & topK & qbs (coded in the code), and save to performance pickle:

python construct_and_search_faiss.py --dbname SIFT1M --index_key IVF1024,Flat --mode run_all --nthreads 16 --nruns 3 --nprobe_max 64 --use_gpu 0 --perf_df_path perf_df.pickle

"""

from __future__ import print_function
import os
import sys
import time
import numpy as np
import re
import faiss
import pickle
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from utils import mmap_fvecs, mmap_bvecs, read_deep_fbin, read_deep_ibin, ivecs_read, \
    mmap_bvecs_SBERT, read_spacev_int8bin, print_recall, calculate_recall
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--faiss_index_path', type=str, default="../data/CPU_Faiss_indexes", help='path to output constructed index')
parser.add_argument('--dbname', type=str, default='SIFT100M', help="dataset name, e.g., SIFT100M")
parser.add_argument('--index_key', type=str, default='IVF1024,PQ16', help="index parameters, e.g., IVF1024,PQ16 or OPQ16,IVF1024,PQ16")

parser.add_argument('--mode', type=str, default='run_one', choices=['run_one', 'run_all', 'energy'], help="run one experiment or run all configs")
parser.add_argument('--nthreads', type=int, default=None, help="(For CPU only) number of threads, if not set, use the max")
parser.add_argument('--use_gpu', type=int, default=0, help="0 -> CPU, 1 -> GPU")

# For Mode "run_one" / "energy"
parser.add_argument('--topK', type=int, default=100, help="return topK most similar vector, related to recall, e.g., R@10=50perc or R@100=80perc")
parser.add_argument('--qbs', type=int, default=1, help="query batch size")
parser.add_argument('--parametersets', type=str, default='nprobe=1', help="a string of nprobes, e.g., 'nprobe=1 nprobe=32'")

# For Mode "run_all"
parser.add_argument('--perf_df_path', type=str, default='perf_df.pickle')
parser.add_argument('--nruns', type=int, default=3, help="number of runs")
parser.add_argument('--nprobe_max', type=int, default=None, help="max nprobe in mode run_all")

args = parser.parse_args()


#################################################################
# Training
#################################################################


def choose_train_size(index_key):

    # some training vectors for PQ and the PCA
    n_train = 256 * 1000

    if "IVF" in index_key:
        matches = re.findall('IVF([0-9]+)', index_key)
        ncentroids = int(matches[0])
        n_train = max(n_train, 100 * ncentroids)
    return n_train


def get_trained_index():
    filename = "%s/%s_%s_trained.index" % (
        index_dir, dbname, index_key)

    if not os.path.exists(filename):
        index = faiss.index_factory(d, index_key)

        n_train = choose_train_size(index_key)

        xtsub = xt[:n_train]
        print("Keeping %d train vectors" % xtsub.shape[0])
        # make sure the data is actually in RAM and in float
        xtsub = xtsub.astype('float32').copy()
        index.verbose = True

        t0 = time.time()
        index.train(xtsub)
        index.verbose = False
        print("train done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
    return index

def rate_limited_imap(f, l):
    'a thread pre-processes the next element'
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i, ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()

def matrix_slice_iterator(x, bs):
    " iterate over the lines of x in blocks of size bs"
    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    return rate_limited_imap(
        lambda i01: x[i01[0]:i01[1]].astype('float32').copy(),
        block_ranges)


def get_populated_index():

    filename = "%s/%s_%s_populated.index" % (
        index_dir, dbname, index_key)

    if not os.path.exists(filename):
        index = get_trained_index()
        i0 = 0
        t0 = time.time()
        for xs in matrix_slice_iterator(xb, 100000):
            i1 = i0 + xs.shape[0]
            print('\radd %d:%d, %.3f s' % (i0, i1, time.time() - t0), end=' ')
            sys.stdout.flush()
            index.add(xs)
            i0 = i1
        print()
        print("Add done in %.3f s" % (time.time() - t0))
        print("storing", filename)
        faiss.write_index(index, filename)
    else:
        print("loading", filename)
        index = faiss.read_index(filename)
    return index


#################################################################
# Perform searches
#################################################################

    
dbname = args.dbname
index_key = args.index_key
topK = args.topK
parametersets = args.parametersets.split() # split nprobe argument string by space

nprobe_max = args.nprobe_max
nruns = args.nruns
perf_df_path = args.perf_df_path
index_dir = os.path.join(args.faiss_index_path, '{}_{}'.format(dbname, index_key))

if not os.path.isdir(index_dir):
    print("%s does not exist, creating it" % index_dir)
    os.mkdir(index_dir)

nthreads = args.nthreads
if args.nthreads:
    faiss.omp_set_num_threads(args.nthreads)

print("Preparing dataset", dbname)

if dbname.startswith('SIFT'):
    # SIFT1M to SIFT1000M
    dbsize = int(dbname[4:-1])
    
    dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/bigann'
    xb = mmap_bvecs(os.path.join(dataset_dir, 'bigann_base.bvecs'))
    xq = mmap_bvecs(os.path.join(dataset_dir, 'bigann_query.bvecs'))
    gt = ivecs_read(os.path.join(dataset_dir, 'gnd/idx_%dM.ivecs' % dbsize))

    # trim xb to correct size
    xb = xb[:dbsize * 1000 * 1000]

    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)
    gt = np.array(gt, dtype=np.int32)

elif dbname.startswith('Deep'):

    assert dbname[:4] == 'Deep' 
    assert dbname[-1] == 'M'
    dbsize = int(dbname[4:-1]) # in million
    dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/deep1b'
    xb = read_deep_fbin(os.path.join(dataset_dir, 'base.1B.fbin'))
    xq = read_deep_fbin(os.path.join(dataset_dir, 'query.public.10K.fbin'))
    # xt = read_deep_fbin('deep1b/learn.350M.fbin')

    # trim xb to correct size
    xb = xb[:dbsize * 1000 * 1000]
    gt = read_deep_ibin('/mnt/scratch/wenqi/Faiss_experiments/deep1b/gt_idx_{}M.ibin'.format(dbsize))

    # Wenqi: load xq to main memory and reshape
    xq = xq.astype('float32').copy()
    xq = np.array(xq, dtype=np.float32)
    
elif dbname.startswith('SBERT1M'):
    dbsize = 1
    # assert dbname[:5] == 'SBERT' 
    # assert dbname[-1] == 'M'
    # dbsize = int(dbname[5:-1]) # in million
    
    dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/sbert'
    xb = mmap_bvecs_SBERT(os.path.join(dataset_dir, 'sbert1M.fvecs'), num_vec=int(dbsize * 1e6))
    # xb = mmap_bvecs_SBERT(os.path.join(dataset_dir, 'sbert3B.fvecs'), num_vec=int(dbsize * 1e6))
    xq = mmap_bvecs_SBERT(os.path.join(dataset_dir, 'query_10K.fvecs'), num_vec=10 * 1000)
    gt = read_deep_ibin(os.path.join(dataset_dir, 'gt_idx_%dM.ibin' % dbsize))

    # trim to correct size
    xb = xb[:dbsize * 1000 * 1000]
    
elif dbname.startswith('SPACEV'):
    """  
    >>> vec_count = struct.unpack('i', fvec.read(4))[0]
    >>> vec_count
    1402020720
    >>> vec_dimension = struct.unpack('i', fvec.read(4))[0]
    >>> vec_dimension
    100
    """
    dbsize = int(dbname[6:-1])
    dataset_dir = '/mnt/scratch/wenqi/Faiss_experiments/SPACEV'
    xb = read_spacev_int8bin(os.path.join(dataset_dir, 'vectors_all.bin'))
    xq = read_spacev_int8bin(os.path.join(dataset_dir, 'query_10K.bin'))
    xq = xq.astype('float32').copy()

    # trim xb to correct size
    xb = xb[:dbsize * 1000 * 1000]
    gt = read_deep_ibin(os.path.join(dataset_dir, 'gt_idx_%dM.ibin' % dbsize))
    
else:
    print('unknown dataset', dbname, file=sys.stderr)
    sys.exit(1)

nq, d = xq.shape
assert gt.shape[0] == nq
xt = xb

index = get_populated_index()
if args.use_gpu:
    res = faiss.StandardGpuResources()  # use a single GPU
    index = faiss.index_cpu_to_gpu(res, 0, index)
    ps = faiss.GpuParameterSpace()
else:
    ps = faiss.ParameterSpace()

ps.initialize(index)

# a static C++ object that collects statistics about searches
ivfpq_stats = faiss.cvar.indexIVFPQ_stats
ivf_stats = faiss.cvar.indexIVF_stats

if args.mode == 'run_one':

    for param in parametersets:
        print(param, '\t', end=' ')
        sys.stdout.flush()
        if index_key != 'Flat':
            ps.set_index_parameters(index, param)

        I = np.empty((nq, topK), dtype='int32')
        D = np.empty((nq, topK), dtype='float32')

        ivfpq_stats.reset()
        ivf_stats.reset()

        latency_per_batch = []
        t0 = time.time()

        i0 = 0
        while i0 < nq:
            t_batch_start = time.time()
            if i0 + args.qbs < nq:
                i1 = i0 + args.qbs
            else:
                i1 = nq
            Di, Ii = index.search(xq[i0:i1], topK)
            I[i0:i1] = Ii
            D[i0:i1] = Di
            i0 = i1
            t_batch_end = time.time()
            latency_per_batch.append(t_batch_end - t_batch_start)

        t1 = time.time()

        print_recall(I, gt)
        print("\nQPS = {}".format(nq / (t1 - t0)))
        latency_per_batch_ms = np.array(latency_per_batch) * 1000
        print("Median batch latency: {:.3f} ms".format(np.median(latency_per_batch_ms)))
        #print("%8.3f  " % ((t1 - t0) * 1000.0 / nq), end=' ms')
        # print("%5.2f" % (ivfpq_stats.n_hamming_pass * 100.0 / ivf_stats.ndis))
elif args.mode == 'run_all':
                
    """ Note: edit here to change the grid search parameters """
    batch_size_list = [1, 2, 4, 8, 16, 32, 10000]
    # batch_size_list = [1]
    nprobe_list = [1]
    while True:
        if nprobe_max is not None and nprobe_list[-1] >= nprobe_max:
            break
        nprobe_list.append(nprobe_list[-1] * 2)

    key_columns = ['dataset', 'max_cores', 'batch_size', 'nprobe']
    result_columns = ['recall_1', 'recall_10', 'latency_ms_per_batch', 'qps']
    columns = key_columns + result_columns

    if os.path.exists(perf_df_path): # load existing
        df = pd.read_pickle(perf_df_path)
        assert len(df.columns.values) == len(columns)
        for col in columns:
            assert col in df.columns.values
        print("Performance dataframe loaded")
    else:
        print(f"Performance dataframe does not exist, create a new one: {perf_df_path}")
        df = pd.DataFrame(columns=columns)
    pd.set_option('display.expand_frame_repr', False)

    for batch_size in batch_size_list:
        for nprobe in nprobe_list:

            latency_ms_per_batch = []
            qps_list = []

            for run in range(nruns):
                print("batch_size={}, nprobe={}".format(batch_size, nprobe))
                ps.set_index_parameters(index, "nprobe={}".format(nprobe))

                I = np.empty((nq, topK), dtype='int32')
                D = np.empty((nq, topK), dtype='float32')

                ivfpq_stats.reset()
                ivf_stats.reset()

                latency_per_batch = []
                t0 = time.time()

                i0 = 0
                while i0 < nq:
                    t_batch_start = time.time()
                    if i0 + batch_size < nq:
                        i1 = i0 + batch_size
                    else:
                        i1 = nq
                    Di, Ii = index.search(xq[i0:i1], topK)
                    I[i0:i1] = Ii
                    D[i0:i1] = Di
                    i0 = i1
                    t_batch_end = time.time()
                    latency_per_batch.append(t_batch_end - t_batch_start)

                t1 = time.time()

                print_recall(I, gt)
                recall_1 = calculate_recall(I, gt, 1)
                recall_10 = calculate_recall(I, gt, 10)

                qps_this_run = nq / (t1 - t0)
                qps_list.append(qps_this_run)
                print("\nQPS = {}".format(nq / (t1 - t0)))

                latency_ms_per_batch_this_run = np.array(latency_per_batch) * 1000
                if len(latency_ms_per_batch_this_run) > 2: # remove first and last batch latency
                    latency_ms_per_batch_this_run = latency_ms_per_batch_this_run[1:-1]
                latency_ms_per_batch.extend(latency_ms_per_batch_this_run)
                print("Median batch latency: {:.3f} ms".format(np.median(latency_ms_per_batch_this_run)))

            key_values = {
                'dataset': dbname,
                'max_cores': nthreads,
                'batch_size': batch_size,
                'nprobe': nprobe
            }
            

            idx = df.index[(df['dataset'] == dbname) & \
                            (df['max_cores'] == nthreads) & \
                            (df['batch_size'] == batch_size) & \
                            (df['nprobe'] == nprobe)
                            ]
            if len(idx) > 0:
                print("Warning: duplicate entry found, deleting the old entry:")
                print(df.loc[idx])
                df = df.drop(idx)
            print(f"Appending new entry:")
            new_entry = {**key_values, 'recall_1': recall_1, 'recall_10': recall_10, 'latency_ms_per_batch': latency_ms_per_batch, 'qps': qps_list}
            print(new_entry)
            df = df.append(new_entry, ignore_index=True)
    
    df.to_pickle(args.perf_df_path, protocol=4)

elif args.mode == 'energy':

    print("Started running search for energy consumption measurement...")
    while True:
        for param in parametersets:
            print(param, '\t', end=' ')
            sys.stdout.flush()
            if index_key != 'Flat':
                ps.set_index_parameters(index, param)

            I = np.empty((nq, topK), dtype='int32')
            D = np.empty((nq, topK), dtype='float32')

            ivfpq_stats.reset()
            ivf_stats.reset()

            i0 = 0
            while i0 < nq:
                if i0 + args.qbs < nq:
                    i1 = i0 + args.qbs
                else:
                    i1 = nq
                Di, Ii = index.search(xq[i0:i1], topK)
                I[i0:i1] = Ii
                D[i0:i1] = Di
                i0 = i1