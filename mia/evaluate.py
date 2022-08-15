import json
import logging
import time 
import os
import mkl
mkl.set_num_threads(24)

import numpy as np
import pandas as pd
from pomegranate import BayesianNetwork
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score, precision_score, recall_score, roc_auc_score, silhouette_score, matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

from scipy.stats import wasserstein_distance 
from mia.constants import CATEGORICAL, CONTINUOUS, ORDINAL

LOGGER = logging.getLogger(__name__)


_MODELS = {
    'binary_classification': [
        {
            'class': DecisionTreeClassifier,
            'kwargs': {
                'max_depth': 20, 
                # 'n_jobs': -1
                # 'class_weight': 'balanced',
            }
        },
        {
            'class': AdaBoostClassifier,
            'kwargs': {
                # 'n_jobs': -1
                # 'class_weight': 'balanced',
            }

        },
        {
            'class': LogisticRegression,
            'kwargs': {
                'solver': 'lbfgs',
                'n_jobs': -1,
                # 'class_weight': 'balanced',
                'max_iter': 50
            }
        },
        {
            'class': MLPClassifier,
            'kwargs': {
                'hidden_layer_sizes': (50, ),
                'max_iter': 50
            },
        }
    ],
    'multiclass_classification': [
        {
            'class': DecisionTreeClassifier,
            'kwargs': {
                'max_depth': 30,
                'class_weight': 'balanced',
            }
        },
        {
            'class': MLPClassifier,
            'kwargs': {
                'hidden_layer_sizes': (100, ),
                'max_iter': 50
            },
        }
    ],
    'clustering': [
        {
            'class': KMeans, 
            'kwargs': {
                'n_clusters': 2,
                'n_jobs': -1,
            }
        }
    ],
    'regression': [
        {
            'class': LinearRegression,
        },
        {
            'class': MLPRegressor,
            'kwargs': {
                'hidden_layer_sizes': (50, ),
                'max_iter': 50
            },
        }
    ],
    'regression2': [
        {
            'class': LinearRegression,
        },
        {
            'class': MLPRegressor,
            'kwargs': {
                'hidden_layer_sizes': (50, ),
                'max_iter': 50,
                'learning_rate_init' : 0.1,
            },
        }
    ]


}


class FeatureMaker:

    def __init__(self, metadata, label_column='label', label_type='int', sample=50000):
        self.columns = metadata['columns']
        self.label_column = label_column
        self.label_type = label_type
        self.sample = sample
        self.encoders = dict()

    def make_features(self, data):
        data = data.copy()
        np.random.shuffle(data)
        data = data[:self.sample]

        features = []
        labels = []

        for index, cinfo in enumerate(self.columns):
            col = data[:, index]
            if cinfo['name'] == self.label_column:
                if self.label_type == 'int':
                    labels = col.astype(int)
                elif self.label_type == 'float':
                    labels = col.astype(float)
                else:
                    assert 0, 'unkown label type'
                continue

            if cinfo['type'] == CONTINUOUS:
                cmin = cinfo['min']
                cmax = cinfo['max']
                if cmin >= 0 and cmax >= 1e3:
                    feature = np.log(np.maximum(col, 1e-2))

                else:
                    feature = (col - cmin) / (cmax - cmin) * 5

            elif cinfo['type'] == ORDINAL:
                feature = col

            else:
                if cinfo['size'] <= 2:
                    feature = col

                else:
                    encoder = self.encoders.get(index)
                    col = col.reshape(-1, 1)
                    if encoder:
                        feature = encoder.transform(col)
                    else:
                        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        self.encoders[index] = encoder
                        feature = encoder.fit_transform(col)

            features.append(feature)

        features = np.column_stack(features)

        return features, labels


def _prepare_ml_problem(train, test, metadata, clustering=False): 
    fm = FeatureMaker(metadata)
    x_train, y_train = fm.make_features(train)
    x_test, y_test = fm.make_features(test)
    if clustering:
        model = _MODELS["clustering"]
    else:
        model = _MODELS[metadata['problem_type']]
    return x_train, y_train, x_test, y_test, model


def _evaluate_multi_classification(train, test, metadata):
    """Score classifiers using f1 score and the given train and test data.

    Args:
        x_train(numpy.ndarray):
        y_train(numpy.ndarray):
        x_test(numpy.ndarray):
        y_test(numpy):
        classifiers(list):

    Returns:
        pandas.DataFrame
    """
    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata)

    performance = []
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        LOGGER.info('Evaluating using multiclass classifier %s', model_repr)
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)

        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average='macro')
        micro_f1 = f1_score(y_test, pred, average='micro')

        performance.append(
            {
                "name": model_repr,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1
            }
        )

    return pd.DataFrame(performance)


def _evaluate_binary_classification(train, test, metadata):
    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata)
    performance = []
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        LOGGER.info('Evaluating using binary classifier %s', model_repr)
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
            pred_prob = np.array([1.] * len(x_test))

        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            pred_prob = model.predict_proba(x_test)


        acc = accuracy_score(y_test, pred)
        binary_f1 = f1_score(y_test, pred, average='binary')
        macro_f1 = f1_score(y_test, pred, average='macro')
        report = classification_report(y_test, pred, output_dict=True)
        weighted_f1 = report['0']['f1-score'] * report['1']['support']/len(y_test) + report['1']['f1-score'] * report['0']['support']/len(y_test)

        mcc = matthews_corrcoef(y_test, pred)

        precision = precision_score(y_test, pred, average='binary')
        recall = recall_score(y_test, pred, average='binary')
        size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
        rest_label = set(range(size)) - set(unique_labels)
        tmp = []
        j = 0
        for i in range(size):
            if i in rest_label:
                tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
            else:
                try:
                    tmp.append(pred_prob[:,[j]])
                except:
                    tmp.append(pred_prob[:, np.newaxis])
                j += 1
        roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))

        performance.append(
            {
                "name": model_repr,
                "accuracy": acc,
                "binary_f1": binary_f1,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "matthews_corrcoef": mcc, 
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc
            }
        )
    
    return pd.DataFrame(performance)

def _evaluate_cluster(train, test, metadata):

    x_train, y_train, x_test, y_test, kmeans = _prepare_ml_problem(train, test, metadata, clustering=True)
 
    model_class = kmeans[0]['class']
    model_repr = model_class.__name__
    model = model_class(n_clusters = metadata['columns'][-1]["size"])
    LOGGER.info('Evaluating using multiclass classifier %s', model_repr)
    unique_labels = np.unique(y_train)

    if len(unique_labels) == 1:
        pred = [unique_labels[0]] * len(x_test)

    else:
        try:
            model.fit(x_train)
            predicted_label = model.predict(x_test)
        except:
            x_train = x_train.astype(np.float32)
            model.fit(x_train)

            x_test = x_test.astype(np.float32)
            predicted_label = model.predict(x_test)
        try:
            pred = silhouette_score(x_test, predicted_label, metric='euclidean', sample_size=100)     
        except:
            pred = 0
    return pred


def _evaluate_regression(train, test, metadata):
    x_train, y_train, x_test, y_test, regressors = _prepare_ml_problem(train, test, metadata)

    performance = []
    if metadata["problem_type"] == "regression":
        y_train = np.log(np.clip(y_train, 1, 20000))
        y_test = np.log(np.clip(y_test, 1, 20000))
    else:
        y_train = np.log(np.clip(y_train, 1, None))
        y_test = np.log(np.clip(y_test, 1, None))
    for model_spec in regressors:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        LOGGER.info('Evaluating using regressor %s', model_repr)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        r2 = r2_score(y_test, pred)
        if r2 < -1:
            r2 = -1.

        performance.append(
            {
                "name": model_repr,
                "r2": r2,
            }
        )

    return pd.DataFrame(performance)


def _evaluate_gmm_likelihood(train, test, metadata, components=[10, 30]):
    results = list()
    for n_components in components:
        gmm = GaussianMixture(n_components, covariance_type='diag')
        LOGGER.info('Evaluating using %s', gmm)
        gmm.fit(test)
        l1 = gmm.score(train)

        gmm.fit(train)
        l2 = gmm.score(test)

        results.append({
            "name": repr(gmm),
            "syn_likelihood": l1,
            "test_likelihood": l2,
        })

    return pd.DataFrame(results)


def _mapper(data, metadata):
    data_t = []
    for row in data:
        row_t = []
        for id_, info in enumerate(metadata['columns']):
            row_t.append(info['i2s'][int(row[id_])])

        data_t.append(row_t)

    return data_t

def change_data_for_dist(data, meta):
    values =[]
    for id_, info in enumerate(meta['columns']):
        if info['type'] == "categorical":
            current = np.zeros([len(data), info['size']])
            idx = data[:, id_].astype(int)
            current[np.arange(len(data)), idx] = 1
            values.append(current)
        else:
            values += [data[:, id_].reshape([-1, 1])]

    return np.concatenate(values, axis=1)


def _compute_distance(train, syn, metadata, sample=300):
    mask_d = np.zeros(len(metadata['columns']))

    for id_, info in enumerate(metadata['columns']):
        if info['type'] in [CATEGORICAL, ORDINAL]:
            mask_d[id_] = 1
        else:
            mask_d[id_] = 0

    std = np.std(train, axis=0) + 1e-6

    dis_all = []
    for i in range(min(sample, len(train))):
        current = syn[i]
        distance_d = (train - current) * mask_d > 0
        distance_d = np.sum(distance_d, axis=1)

        distance_c = (train - current) * (1 - mask_d) / 2 / std
        distance_c = np.sum(distance_c ** 2, axis=1)
        distance = np.sqrt(np.min(distance_c + distance_d))
        dis_all.append(distance)

    return np.mean(dis_all)

def _compute_EMD(train,syn):
    col = train.shape[1]
    result = []
    for i in range(col):
        result.append(wasserstein_distance(train[:,i], syn[:,i]))
    return sum(result) / len(result)


_EVALUATORS = {
    'binary_classification': _evaluate_binary_classification,
    'multiclass_classification': _evaluate_multi_classification,
    'regression': _evaluate_regression,
    'clustering': _evaluate_cluster,
}


def compute_scores(train, test, synthesized_data, metadata, args):
    
    evaluator = _EVALUATORS[metadata['problem_type']]

    scores = pd.DataFrame()

    if args.baseline:
        score = evaluator(synthesized_data, test, metadata) 
        score['distance'] = _compute_distance(train, synthesized_data, metadata, sample=synthesized_data.shape[0])

        train_change, _, synthesized_change = change_data_for_dist(train, metadata), change_data_for_dist(test, metadata), change_data_for_dist(synthesized_data, metadata)
        score['EMD'] = _compute_EMD(train_change, synthesized_change)
        scores = pd.concat([scores, score], ignore_index=True)

    else:
        for i in range(args.test_iter):
            print("start {}-th test".format(i))

            score = evaluator(synthesized_data, test, metadata) 
            score['distance'] = _compute_distance(train, synthesized_data, metadata, sample=synthesized_data.shape[0])
            score['test_iter'] = i

            scores = pd.concat([scores, score], ignore_index=True)
        scores = scores.groupby(['test_iter']).mean()


    return scores
