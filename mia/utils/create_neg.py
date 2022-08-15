import numpy as np

def create_neg(train_data):
    for i in range(train_data.shape[1]):
        shuffled = sorted(train_data[:,i], key=lambda k: np.random.random())
        train_data[:,i] = shuffled
    return train_data