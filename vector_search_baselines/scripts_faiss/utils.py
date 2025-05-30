import sys
import time
import numpy as np
import struct
import os

def calculate_recall(I, gt, k):
    """
    I: ANN search result. numpy array of shape (nq, ?)
    gt: numpy array of shape (>=nq, ?)
    """
    assert I.shape[1] >= k
    assert gt.shape[1] >= k
    nq = I.shape[0]
    total_intersect = 0
    for i in range(nq):
        n_intersect = np.intersect1d(I[i, :k], gt[i, :k], assume_unique=False, return_indices=False).shape[0]
        total_intersect += n_intersect
    return total_intersect / (nq * k)

def print_recall(I, gt): 
    """
    print recall (depends on the shape of I, return 1/10/100)
    """
    k_max = I.shape[1]
    if k_max >= 100:
        k_set = [1, 10, 100]
        print(' ' * 4, '\t', 'R@1    R@10   R@100')
    elif k_max >= 10:
        k_set = [1, 10]
        print(' ' * 4, '\t', 'R@1    R@10')
    else:
        k_set = [1]
        print(' ' * 4, '\t', 'R@1')
    for k in k_set:
        print("{:.4f}".format(calculate_recall(I, gt, k)), end=' ')

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def ivecs_read(fname):
    """
    Used to read the ground truth file in ivecs format
    """
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    # Example format of ground truth (for 10000 query vectors):
    #   1000(topK), [1000 ids]
    #   1000(topK), [1000 ids]
    #        ...     ...
    #   1000(topK), [1000 ids]
    # 10000 rows in total, 10000 * 1001 elements, 10000 * 1001 * 4 bytes
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    """
    Used to read the vector file in fvecs format
    """
    return ivecs_read(fname).view('float32')

def load_sift1M(fpath='/mnt/scratch/wenqi/Faiss_experiments/sift1M'):
    """
    Read the sift1M dataset, note it's not equivalent to the first 1M of the SIFT1B dataset
    """
    print("Loading sift1M...", end='', file=sys.stderr)
    xt = fvecs_read(os.path.join(fpath, "sift_learn.fvecs"))
    xb = fvecs_read(os.path.join(fpath, "sift_base.fvecs"))
    xq = fvecs_read(os.path.join(fpath, "sift_query.fvecs"))
    gt = ivecs_read(os.path.join(fpath, "sift_groundtruth.ivecs"))
    print("done", file=sys.stderr)

    return xb, xq, xt, gt


def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    D, I = index.search(xq, k)  # noqa: E741
    t1 = time.time()

    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)
        i *= 10

    return (t1 - t0) * 1000.0 / nq, recalls

def mmap_bvecs_SBERT(fname, num_vec=int(1e6)):
    """
    SBERT, 384 dim, no header
    """
    d = 384
    x = np.memmap(fname, dtype='float32', mode='r')
    return x.reshape(-1, d)

def read_deep_fbin(filename):
    """
    Read *.fbin file that contains float32 vectors

    All embedding data is stored in .fbin format:
    [num_vectors (uint32), vector_dim (uint32), vector_array (float32)]

    The groundtruth is stored in .ibin format:
    [num_vectors (uint32), vector_dim (uint32), vector_array (int32)]

    https://research.yandex.com/datasets/biganns
    https://pastebin.com/BAf6bM5L
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)

    arr = np.memmap(filename, dtype=np.float32, offset=8, mode='r')
    return arr.reshape(nvecs, dim)

def read_deep_ibin(filename, dtype='int32'):
    """
    Read *.ibin file that contains int32, uint32 or int64 vectors

    All embedding data is stored in .fbin format:
    [num_vectors (uint32), vector_dim (uint32), vector_array (float32)]

    The groundtruth is stored in .ibin format:
    [num_vectors (uint32), vector_dim (uint32), vector_array (int32)]

    https://research.yandex.com/datasets/biganns
    https://pastebin.com/BAf6bM5L
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
    arr = np.fromfile(filename, dtype=dtype, offset=8)
    return arr.reshape(nvecs, dim)

def read_spacev_int8bin(filename):
    """
    Read *.fbin file that contains int8 vectors

    All embedding data is stored in .bin format:
    [num_vectors (uint32), vector_dim (uint32), vector_array (int8)]

    The groundtruth is stored in .ibin format:
    [num_vectors (uint32), vector_dim (uint32), vector_array (int32)]
    
	>>> vec_count = struct.unpack('i', fvec.read(4))[0]
	>>> vec_count
	1402020720
	>>> vec_dimension = struct.unpack('i', fvec.read(4))[0]
	>>> vec_dimension
	100

    https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B
    """
    with open(filename, "rb") as f:
        # nvecs_total means the total number of vectors across all partitions
        nvecs_total, dim = np.fromfile(f, count=2, dtype=np.int32) 

    arr = np.memmap(filename, dtype=np.int8, offset=8, mode='r')
    return arr.reshape(nvecs_total, dim)

def write_deep_fbin(filename, vecs):
    """ Write an array of float32 vectors to *.fbin file
    Args:s
        :param filename (str): path to *.fbin file
        :param vecs (numpy.ndarray): array of float32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('float32').flatten().tofile(f)
 
        
def write_deep_ibin(filename, vecs, dtype='int32'):
    """ Write an array of int32 or int64 vectors to *.ibin file
    Args:
        :param filename (str): path to *.ibin file
        :param vecs (numpy.ndarray): array of int32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        if dtype == 'int32':
            vecs.astype('int32').flatten().tofile(f)
        elif dtype == 'uint32':
            vecs.astype('uint32').flatten().tofile(f)
        elif dtype == 'int64':
            vecs.astype('int64').flatten().tofile(f)
        else:
            print("Unsupported datatype ", dtype)
            raise ValueError