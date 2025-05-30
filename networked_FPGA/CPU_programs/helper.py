import pickle
import os

def save_obj(obj, dirc, name):
    # note use "dir/" in dirc
    # can replace pickle.HIGHEST_PROTOCOL by 4 for py37, 5 supported only for python (not sure 3.8 or 3.9)
    if '.pkl' not in name:
        name = name + '.pkl'
    with open(os.path.join(dirc, name), 'wb') as f:
        pickle.dump(obj, f, protocol=4) # for py37,pickle.HIGHEST_PROTOCOL=4


def load_obj(dirc, name):
    if '.pkl' not in name:
        name = name + '.pkl'
    with open(os.path.join(dirc, name), 'rb') as f:
        return pickle.load(f) # for py37,pickle.HIGHEST_PROTOCOL=4
