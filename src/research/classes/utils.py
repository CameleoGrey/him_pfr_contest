
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
import pickle
import numpy as np
from copy import deepcopy
import pandas as pd
import os

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

# universal function to serialize object
def save(obj, path, verbose=True):
    if verbose:
        print("Saving object to {}".format(path))

    with open(path, "wb") as obj_file:
        #joblib.dump(obj, obj_file)
        pickle.dump( obj, obj_file, protocol=pickle.HIGHEST_PROTOCOL )

    if verbose:
        print("Object saved to {}".format(path))
    pass

# universal function to deserialize on object
def load(path, verbose=True):
    if verbose:
        print("Loading object from {}".format(path))
    with open(path, "rb") as obj_file:
        #obj = joblib.load(obj_file)
        obj = pickle.load(obj_file)
    if verbose:
        print("Object loaded from {}".format(path))
    return obj


def precision_at_k(y_true, y_pred, k=12):
    """ Computes Precision at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Precision at k
    """
    intersection = np.intersect1d(y_true, y_pred[:k])
    score = len(intersection) / k
    return score


def rel_at_k(y_true, y_pred, k=12):
    """ Computes Relevance at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Relevance at k
    """
    if y_pred[k - 1] in y_true:
        return 1
    else:
        return 0


def average_precision_at_k(y_true, y_pred, k=12):
    """ Computes Average Precision at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Average Precision at k
    """
    ap = 0.0
    for i in range(1, k + 1):
        ap += precision_at_k(y_true, y_pred, i) * rel_at_k(y_true, y_pred, i)

    score = ap / min(k, len(y_true))

    return score


def mean_average_precision(y_true, y_pred, k=12):
    """ Computes MAP at k

    Parameters
    __________
    y_true: np.array
            2D Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            2D Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           MAP at k
    """

    score = np.mean([average_precision_at_k(gt, pred, k) \
                    for gt, pred in zip(y_true, y_pred)])

    return score
