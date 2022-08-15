import json
import logging
import os
import numpy as np

from mia.constants import CATEGORICAL, ORDINAL


LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)
    if loader == np.load:
        return loader(local_path, allow_pickle=True)
    return loader(local_path)


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns


def load_dataset_test(name, benchmark=False, num=0):
    randomSeed = 2021
    np.random.seed(randomSeed)

    LOGGER.info('Loading dataset %s', name)
    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    categorical_columns, ordinal_columns = _get_columns(meta)

    train = data['train']
    test = data['test']

    if num != 0 :
        data = np.concatenate([train, test])
        np.random.shuffle(data)
        data_size = num
        if data_size <= 500:
            data_size = 500
        sample_size = data_size
        data_idx = np.random.randint(low=data.shape[0], size=sample_size)
        data = data[data_idx]

        train = data[:int(sample_size*0.8)]
        test = data[int(sample_size*0.8):]
        
    if benchmark:
        return train, test, meta, categorical_columns, ordinal_columns

    return train, categorical_columns, ordinal_columns

def load_dataset(name, benchmark=False):

    LOGGER.info('Loading dataset %s', name)
    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    categorical_columns, ordinal_columns = _get_columns(meta)

    train = data['train']
    test = data['test']
    if benchmark:
        return train, test, meta, categorical_columns, ordinal_columns

    return train, categorical_columns, ordinal_columns



