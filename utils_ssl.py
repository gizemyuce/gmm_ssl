import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import linalg as LA
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.semi_supervised import SelfTrainingClassifier
import openml
import pickle
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import PredefinedSplit, GridSearchCV

# GLOBAL_SOLVER = "saga"
GLOBAL_SOLVER = "lbfgs"
GLOBAL_MAX_ITER = 100
# GLOBAL_MAX_ITER = 1000


def plot(x_1_labeled, x_2_labeled, x_unlabeled, W_OPT= np.ones(2)/np.sqrt(2), GAMMA=None , LIM=10):
  _, ax = plt.subplots(figsize=(10,10))
  ax.set_aspect('equal', adjustable='box')
  ax.set_xlim(-LIM, LIM)
  ax.set_ylim(-LIM, LIM)


#   # opt & margin
#   _x = np.linspace(-LIM, LIM, 100)
#   _y = - W_OPT[0] * _x / W_OPT[1]
#   ax.plot(_x, _y, c='black')

  if GAMMA is not None:
    ax.plot(_x, _y + GAMMA, c='grey')
    ax.plot(_x, _y - GAMMA, c='grey')


    ax.scatter(x_unlabeled[:, 0], x_unlabeled[:, 1], marker='.', c='lightblue', s=5, label='unl')
    ax.scatter(x_1_labeled[:, 0],x_1_labeled[:, 1], marker='+', c='blue', s=20, label='0')
    ax.scatter(x_2_labeled[:, 0], x_2_labeled[:, 1], marker='_', c='red', s=20, label='1')
    plt.legend()
    return plt

def expectation_maximization_kmeans_init(x_unlabeled):
#     print(x_unlabeled[0], x_unlabeled.shape)
    gm = GaussianMixture(n_components=2, random_state=0, covariance_type='diag').fit(x_unlabeled)
    return gm.means_

def expectation_maximization_supervised_init(x_unlabeled,x_1_labeled, x_2_labeled ):
    mean_1 = np.mean(x_1_labeled, axis=0)
    mean_2 = np.mean(x_2_labeled, axis=0)
    means_init = np.append(mean_1[None, :], mean_2[None, :] ,axis=0)
    gm = GaussianMixture(n_components=2, random_state=0, means_init = means_init).fit(x_unlabeled)
    return gm.means_

def get_sup_estimator(x_labelled, y, x_val, y_val):
#     clf = LogisticRegression(random_state=0).fit(x_labelled, y)
#     return clf
    all_xs = np.concatenate((x_labelled, x_val))
    all_ys = np.concatenate((y, y_val))
    val_idxs = np.concatenate((-np.ones(x_labelled.shape[0]), np.zeros(x_val.shape[0])))
    ps = PredefinedSplit(test_fold=val_idxs)
    clf = LogisticRegression(random_state=0, solver=GLOBAL_SOLVER,
                             max_iter=GLOBAL_MAX_ITER)
    C_values = [1., 0.1, 0.01, 0.001]
    grid_search = GridSearchCV(
            clf,
            refit=False,
            param_grid={"C": C_values},
            cv=ps).fit(all_xs, all_ys)
    best_clf = LogisticRegression(random_state=0, solver=GLOBAL_SOLVER,
                                  max_iter=GLOBAL_MAX_ITER,
                                  C=grid_search.best_params_["C"]).fit(x_labelled, y)

    return best_clf


def get_unsup_estimator(x_unlabelled, x_val, y_val):
    em_means = expectation_maximization_kmeans_init(x_unlabelled)
    clf_em = LogisticRegression(random_state=0, solver=GLOBAL_SOLVER,
                             max_iter=GLOBAL_MAX_ITER)
    clf_em.coef_=(em_means[0] - em_means[1]).reshape(1,-1)
    clf_em.intercept_=0
    clf_em.classes_=np.unique(y_val)
    if clf_em.score(x_val, y_val) <0.5:
        clf_em.coef_*=-1
    return clf_em


def get_ssl_estimator(clf_sl, clf_em, lambda_):
    clf_ssl = LogisticRegression(random_state=0)
    w=lambda_*clf_em.coef_ + (1-lambda_)*clf_sl.coef_
    w0=(1-lambda_)*clf_sl.intercept_

    clf_ssl.coef_=w
    clf_ssl.intercept_=w0
    clf_ssl.classes_=clf_sl.classes_
    return clf_ssl

def get_best_lambda(clf_sl, clf_em, x_val, y_val=None):
    score_arr=[]
    best_lambda=0
    best_score=0
    for lambda_ in np.linspace(0,1,1000):
        clf_ssl=get_ssl_estimator(clf_sl, clf_em, lambda_)
        if y_val is not None:
            # Use supervised validation data for selecting best lambda.
            score_curr = clf_ssl.score(x_val, y_val)
        else:
            # Use unsupervised validation data for selecting best lambda.
            score_curr = (clf_ssl.intercept_ + x_val @ clf_ssl.coef_.reshape(-1, 1)) / np.linalg.norm(clf_ssl.coef_)
            score_curr = -score_curr.mean()
        if score_curr > best_score:
            best_lambda=lambda_
            best_score = score_curr
    return best_lambda, best_score

def get_mnist_classes(class1=5, class2=9, n_components=20):
#     data = openml.datasets.get_dataset('mnist_784')
#     x, y, _, _ = data.get_data(target=data.default_target_attribute)
    with open('data/mnist_784.pkl', 'rb') as handle:
                data_dict = pickle.load(handle)
    x, y = data_dict["x"], data_dict["y"]
    x = x.to_numpy()
    y = y.to_numpy(dtype="int")
    scaler = preprocessing.StandardScaler()
    x_subset=np.concatenate((x[y==class1],x[y==class2]) )
    x_scaled=scaler.fit_transform(x_subset)
    y_subset=np.concatenate((y[y==class1], y[y==class2]))

    pca=PCA(n_components=n_components)
    x_trans=pca.fit_transform(x_scaled)
    perm_idx=np.random.permutation(len(x_trans))
    x_trans=x_trans[perm_idx]
    y_subset=y_subset[perm_idx]
    labels = np.unique(y_subset)
    y_subset = np.array([0 if yyy == labels[0] else 1 for yyy in y_subset])
    return x_trans, y_subset

def get_openml_data(dataset_name, n_components=20, balance_data=True):
    with open(f'data/{dataset_name}.pkl', 'rb') as handle:
        data_dict = pickle.load(handle)
    x, y = data_dict["x"], data_dict["y"]
    x = x.to_numpy()
    if "musk" in dataset_name or "Santander" in dataset_name:
        x = x[:, 1:]
    x = np.array(x, dtype="float")
    y = y.to_numpy(dtype="int")
    scaler = preprocessing.StandardScaler()
    x_subset=x
    y_subset=y
    x_scaled=scaler.fit_transform(x_subset)

    # PCA
    pca=PCA(n_components=n_components)
    x_trans=pca.fit_transform(x_scaled)
    perm_idx=np.random.permutation(len(x_trans))
    x_trans=x_trans[perm_idx]
    y_subset=y_subset[perm_idx]

    if balance_data:
        np.random.seed(42)
        x_trans, y_subset, _ = shuffle(x_trans, y_subset)
        neg_idxs, pos_idxs = np.where(y_subset == -1)[0], np.where(y_subset == 1)[0]
        neg_size, pos_size = neg_idxs.shape[0], pos_idxs.shape[0]
        if neg_size < pos_size:
            idx = np.concatenate((neg_idxs, pos_idxs[:neg_size]))
        else:
            idx = np.concatenate((neg_idxs[:pos_size], pos_idxs))
        np.random.seed(42)
        x_trans, y_subset, _ = self.shuffle_data(x_trans[idx], y_subset[idx])

    labels = np.unique(y_subset)
    y_subset = np.array([0 if yyy == labels[0] else 1 for yyy in y_subset])
    return x_trans, y_subset


def get_self_training_estimator(x_labelled, y, x_unlabelled, x_val, y_val):
#     all_xs = np.concatenate((x_labelled, x_unlabelled))
#     all_ys = np.concatenate((y, -np.ones(x_unlabelled.shape[0])))
#     clf = LogisticRegression(random_state=0)
#     self_training_model = SelfTrainingClassifier(clf).fit(all_xs, all_ys)
#     return self_training_model
    ssl_xs = np.concatenate((x_labelled, x_unlabelled))
    ssl_ys = np.concatenate((y, -np.ones(x_unlabelled.shape[0])))
    all_xs = np.concatenate((x_labelled, x_unlabelled, x_val))
    all_ys = np.concatenate((y, -np.ones(x_unlabelled.shape[0]), y_val))
    val_idxs = np.concatenate((-np.ones(x_labelled.shape[0]+x_unlabelled.shape[0]), np.zeros(x_val.shape[0])))
    ps = PredefinedSplit(test_fold=val_idxs)
    clf = LogisticRegression(random_state=0, solver=GLOBAL_SOLVER,
                             max_iter=GLOBAL_MAX_ITER)
    thresh_values = [0.6, 0.7, 0.8, 0.9]
    selfT_clf = SelfTrainingClassifier(clf)
    grid_search = GridSearchCV(
            selfT_clf,
            refit=False,
            param_grid={"threshold": thresh_values},
            cv=ps).fit(all_xs, all_ys)
    best_clf = SelfTrainingClassifier(clf, threshold=grid_search.best_params_["threshold"]).fit(ssl_xs, ssl_ys)
    return best_clf

def get_labelprop_estimator(x_labelled, y, x_unlabelled, x_val, y_val):
    all_xs = np.concatenate((x_labelled, x_unlabelled, x_val))
    all_ys = np.concatenate((y, -np.ones(x_unlabelled.shape[0]), y_val))
    val_idxs = np.concatenate((-np.ones(x_labelled.shape[0]+x_unlabelled.shape[0]), np.zeros(x_val.shape[0])))
    ps = PredefinedSplit(test_fold=val_idxs)
    print(ps.get_n_splits())
    clf = LabelPropagation(kernel="knn", n_jobs=8, max_iter=10000)
    grid_search = GridSearchCV(clf, param_grid={"n_neighbors": [3, 4, 6]}, cv=ps).fit(all_xs, all_ys)
    return grid_search

def shuffle(x, y):
    perm_idx=np.random.permutation(len(x))
    new_x=x[perm_idx]
    new_y=y[perm_idx]
    return new_x, new_y

def get_linear_error(dataset_name):
    return {
        "a9a": 0.1789,
        "vehicleNorm": 0.1415,
        "w8a": 0.0947,
        "nomao": 0.0531,
        "SantanderCustomerSatisfaction": 0.2188,
        "webdata_wXa": 0.1813,
        "sylva_prior": 0.0011,
        "real-sim": 0.0027,
        "riccardo": 0.0007,
        "guillermo": 0.2536,
        "epsilon": 0.0947,
        "jasmine": 0.1867,
        "philippine": 0.2445,
        "madeline": 0.3405,
        "christine": 0.1408,
        "musk": 0.0438,
    }[dataset_name]

def get_UL_error(dataset_name):
    return {
        "a9a": 0.2161,
        "vehicleNorm": 0.1770,
        "jasmine": 0.2469,
        "madeline": 0.3818,
        "philippine": 0.3189,
        "musk": 0.2702,
    }[dataset_name]

def parse_mnist_name(dataset_name):
    d, classes = dataset_name.split("_")
    assert d == "mnist"
    c1, c2 = [int(c) for c in classes.split("v")]
    return c1, c2
