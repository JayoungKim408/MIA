
import numpy as np
from mia.constants import CATEGORICAL, ORDINAL

def compute_distance(syn, querys, metadata, use_std = True):
    mask_d = np.zeros(len(metadata['columns']))

    for id_, info in enumerate(metadata['columns']):
        if info['type'] in [CATEGORICAL, ORDINAL]:
            mask_d[id_] = 1
        else:
            mask_d[id_] = 0

    if use_std:
        std = np.std(syn, axis=0) + 1e-6
    else:
        std = 1

    dis_all = []
    for i in range(len(querys)):
        current = querys[i]
        distance_d = np.abs((syn - current)) * mask_d > 1e-6 
        distance_d = np.sum(distance_d, axis=1) 
        distance_c = (syn - current) * (1 - mask_d) / 2 / std
        distance_c = np.sum(distance_c ** 2, axis=1)
        distance = np.sqrt(np.min(distance_c + distance_d))
        dis_all.append(distance)
    return np.array(dis_all)