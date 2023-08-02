import argparse
import warnings
import mlflow
import numpy as np
import pandas as pd
import openml
import os
import random
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from typing import Tuple, List
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import csv


from utils_ssl import *

os.environ["MLFLOW_TRACKING_USERNAME"] = "exp-07.mlflow-yang.alex"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "alexpwd"
remote_server_uri = "https://exp-07.mlflow-yang.inf.ethz.ch"
mlflow.set_tracking_uri(remote_server_uri)

# mlflow.set_experiment("dataset_stats")
# warnings.filterwarnings("ignore", category=DeprecationWarning)

n_pca_components = None
# max_unlabeled_size = 3000
max_unlabeled_size = None


datasets = [
    "a9a",
    "vehicleNorm",
    "jasmine",
    "madeline",
    "philippine",
    "musk",
    "nomao",
    "SantanderCustomerSatisfaction",

    "mnist_5v9",
    "mnist_0v1",
    "mnist_3v8",
    "mnist_5v6",
    "mnist_1v7",

#     "w8a",
#     "webdata_wXa",
#     "sylva_prior",
#     "gisette",
#     "real-sim",
#     "riccardo",
#     "guillermo",
#     "epsilon",
    # XXX: should ignore datasets below this point
#     "vehicle_sensIT",
]

def train_ul(X, y, data_name=""):
    start_time = time.time()

#     em_means = expectation_maximization_kmeans_init(X)
#     clf_em = LogisticRegression(random_state=0)
#     clf_em.coef_=(em_means[0] - em_means[1]).reshape(1,-1)
#     clf_em.intercept_=0
#     clf_em.classes_=np.unique(y)
#     train_error = clf_em.score(X, y)
#     if train_error < 0.5:
#         train_error = 1 - train_error

    model_class1 = GaussianMixture(n_components=1, covariance_type='spherical', reg_covar=1e-6).fit(X[y == 0])
    model_class2 = GaussianMixture(n_components=1, covariance_type='spherical', reg_covar=1e-6).fit(X[y == 1])
    print(f"[data={data_name}] Time", time.time() - start_time)

    cov = (model_class1.covariances_[0] + model_class2.covariances_[0]) / 2
    nb = LinearDiscriminantAnalysis()
    nb.coef_ = np.array([1/cov * (model_class2.means_[0] - model_class1.means_[0])])
    nb.intercept_ = -0.5 * 1/cov * (model_class2.means_[0].T @ model_class2.means_[0] - model_class1.means_[0].T @ model_class1.means_[0])
    nb.classes_ = np.array([0, 1])
    train_error = (nb.predict(X) != y).mean()

    return train_error


def train_linear_model(X, y, data_name="", solver="saga"):
    start_time = time.time()
    gt = init_prediction_model(solver=solver)
    gt.fit(X, y)
    print(f"[data={data_name}] Time", time.time() - start_time)

    y_pred = gt.predict(X)
    train_error = 1 - gt.score(X, y)
    return train_error


def init_prediction_model(solver, penalty="none", C=0.0, scale_data=True):
    if scale_data:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver=solver, penalty=penalty, C=C, max_iter=1e5, tol=1e-4, n_jobs=4
            ),
        )
    else:
        return LogisticRegression(solver=solver, penalty=penalty, C=C)


if __name__ == "__main__":
    data_stats = {}
    for d in datasets:
        if "mnist" not in d:
            X, y = get_openml_data(dataset_name=d,
                                   n_components=n_pca_components,
                                   balance_data=False)
        else:
            c1, c2 = parse_mnist_name(d)
            X, y = get_mnist_classes(class1=c1, class2=c2, n_components=20)

        if max_unlabeled_size is not None:
            assert max_unlabeled_size <= 3000, "cutoff needs to be such that smallest dataset has enough data"
            X, y = shuffle(X, y)
            X, y = X[:max_unlabeled_size], y[:max_unlabeled_size]

        dataset_name = f"{d}"
        print(dataset_name)
        ul_train_error = train_ul(X, y, dataset_name)
        sl_train_error = train_linear_model(X, y, dataset_name)
        data_stats[dataset_name] = [sl_train_error, ul_train_error, X.shape[1]]

    data_stats = dict(sorted(data_stats.items(), key=lambda item: item[1][1] - item[1][0]))
    if max_unlabeled_size is not None:
        filename = f"dataset_UL_size{max_unlabeled_size}_train_error.txt"
    else:
        filename = "dataset_UL_train_error.txt"
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(["dataset", "SLbayes_train_error",
                         "ULbayes_train_error", "ULbayes_error-SLbayes_error",
                         "d"])
        for k, v, in data_stats.items():
            writer.writerow([k, v[0], v[1], v[1] - v[0], v[2]])
