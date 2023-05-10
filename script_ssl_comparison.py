import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import linalg as LA
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import openml
from tqdm import tqdm
import statistics
import pandas as pd
import seaborn as sns
from utils_ssl import *
import time
import csv
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset.",
    )
    args = parser.parse_args()

    num_repetitions = 20
    n_pca_components = 20
    n_unlabelled = 4000
    n_test_set = 1000
    n_val_set = 1000
    n_labelled_arr=np.arange(2,20)
    # dataset_name = "mnist_5_vs_9"
    dataset_name = args.dataset_name

#     if dataset_name in ["a9a"]:
#         n_labelled_arr=np.arange(10, 25, 1)
#     elif dataset_name in ["madeline", "jasmine", "philippine"]:
#         n_labelled_arr=np.arange(50, 500, 50)
#         n_unlabelled = 2000
#     elif dataset_name in ["musk"]:
#         n_unlabelled = 800
#     elif dataset_name in ["mnist_784"]:
#         n_labelled_arr=np.arange(2,20)
#         n_unlabelled = 4000


    if "mnist" in dataset_name:
        x_trans, y_subset = get_mnist_classes(n_components=n_pca_components)
    else:
        x_trans, y_subset = get_openml_data(dataset_name=dataset_name,
                                            n_components=n_pca_components,
                                            balance_data=False)

    n_unlabelled_arr=[n_unlabelled]
    x_test=x_trans[n_labelled_arr[-1]+n_unlabelled_arr[-1]:]
    y_test=y_subset[n_labelled_arr[-1]+n_unlabelled_arr[-1]:]

    ssl_arr=[]
    theory_arr=[]
    lambda_arr=[]

    sl_arr=[]
    st_arr=[]
    lp_arr=[]
    ul_arr=[]
    sl_diff_arr=[]
    ul_diff_arr=[]
    st_diff_arr=[]
    lp_diff_arr=[]

    n_labelled_arrs=[]
    n_unlabelled_arrs=[]

    x_train=x_trans[:-(n_test_set+n_val_set)]
    y_train=y_subset[:-(n_test_set+n_val_set)]
    x_val=x_trans[-(n_test_set+n_val_set):-n_test_set]
    y_val=y_subset[-(n_test_set+n_val_set):-n_test_set]
    x_test=x_trans[-n_test_set:]
    y_test=y_subset[-n_test_set:]

    print(f"[{dataset_name}] train={x_train.shape[0]} val={x_val.shape[0]} test={x_test.shape[0]}")
    labels = np.unique(y_train)
    print(f"[{dataset_name}] imbalance ratio {y_train[y_train == labels[0]].shape[0] / y_train[y_train == labels[1]].shape[0]}")
    assert len(labels) == 2
    for trial_idx in tqdm(range(num_repetitions)):
        while True:
            x_train, y_train = shuffle(x_train, y_train)
            if len(np.unique(y_train[:min(n_labelled_arr)])) == 2:
                break
            print("Another attempt to get labeled set")

#         n_labelled_arr=np.arange(10, 500, 10)
        n_labelled_arr=np.arange(10, 50, 10)
        for n_labelled in n_labelled_arr:
            for n_unlabelled in n_unlabelled_arr:
                n_unlabelled = n_labelled * 100
                x_labelled = x_train[:n_labelled]
                y_sl = y_train[:n_labelled]
                print(f"[{dataset_name}] imbalance ratio {y_sl[y_sl == 0].shape[0] / y_sl[y_sl == 1].shape[0]}")
                x_labelled, y_sl = shuffle(x_labelled, y_sl)
                x_unlabelled=x_train[n_labelled:n_labelled+n_unlabelled]

                assert x_labelled.shape[0] == n_labelled, f"Not enough samples for labeled set of size {n_labelled}; found only {x_labelled.shape[0]}"
                assert x_unlabelled.shape[0] == n_unlabelled, f"Not enough samples for unlabeled set of size {n_unlabelled}; found only {x_unlabelled.shape[0]}"
                clf_sl=get_sup_estimator(x_labelled, y_sl, x_val, y_val)
                clf_st=get_self_training_estimator(x_labelled=x_labelled,
                                                   y=y_sl,
                                                   x_unlabelled=x_unlabelled,
                                                   x_val=x_val,
                                                   y_val=y_val)
                sl_score=clf_sl.score(x_test, y_test)
                st_score=clf_st.score(x_test, y_test)

                clf_em=get_unsup_estimator(x_unlabelled, x_val, y_val)
                ul_score=clf_em.score(x_test, y_test)

                best_lambda, ssl_score = get_best_lambda(clf_sl, clf_em, x_val, y_val)
                ssl_arr.append(ssl_score)
                lambda_arr.append(best_lambda)
                n_labelled_arrs.append(n_labelled)
                n_unlabelled_arrs.append(n_unlabelled)
                st_arr.append(st_score)
                sl_arr.append(sl_score)
    #             lp_arr.append(lp_score)
                ul_arr.append(ul_score)
                sl_diff_arr.append(ssl_score-sl_score)
                ul_diff_arr.append(ssl_score-ul_score)
                st_diff_arr.append(ssl_score-st_score)
    #             lp_diff_arr.append(ssl_score-lp_score)

    # print(df)
    test_dict={
            'n_labelled':n_labelled_arrs,
            'n_unlabelled':n_unlabelled_arrs,
             'sl_acc':sl_arr,
             'ssl_acc':ssl_arr,
             'ul_acc':ul_arr,
             'st_acc':st_arr,
             'sl_diff_acc':sl_diff_arr,
             'ul_diff_acc':ul_diff_arr,
             'st_diff_acc':st_diff_arr,
             }
    print(f"SL: {sl_arr[-1]}; UL: {ul_arr[-1]}; SSL: {ssl_arr[-1]}; SelfT: {st_arr[-1]}")
    df=pd.DataFrame.from_dict(test_dict)
    sns.set()
    fig,ax=plt.subplots(figsize=(8,6))
    sns.lineplot(data=df, x='n_labelled', y='sl_diff_acc',ax=ax, label='SSL-SL')
    sns.lineplot(data=df, x='n_labelled', y='ul_diff_acc',ax=ax, label='SSL-UL')
    sns.lineplot(data=df, x='n_labelled', y='st_diff_acc',ax=ax, label='SSL-SelfT')
#     sns.lineplot(data=df, x='n_labelled', y='ssl_acc',ax=ax, label='SSL')
#     sns.lineplot(data=df, x='n_labelled', y='sl_acc',ax=ax, label='SL')
#     sns.lineplot(data=df, x='n_labelled', y='ul_acc',ax=ax, label='UL')
#     sns.lineplot(data=df, x='n_labelled', y='st_acc',ax=ax, label='SelfT')
    ax.legend(fontsize=22)
    # plt.yscale('log')

    suffix = int(time.time())
    filename = f"{dataset_name}_{n_pca_components}D_{suffix}"
    plt.savefig(f'/Users/alexandrutifrea/Projects/SSL_lower_bound/gmm_ssl/figures/{filename}.png')
    params = {
        "num_repetitions": num_repetitions,
        "data": dataset_name,
        "n_pca_components": n_pca_components,
        "n_all_train": x_train.shape[0],
        "n_labelled_values": n_labelled_arrs,
        "n_unlabelled_values": n_unlabelled_arrs,
        "n_test_set": n_test_set,
        "n_val_set": n_val_set,
    }

    with open(f"figures/{filename}.csv", "w") as f:
        writer = csv.writer(f)
        for k, v, in params.items():
            writer.writerow([k, v])
        for k, v, in test_dict.items():
            writer.writerow([k, v])
    #     print(params, file=f)
