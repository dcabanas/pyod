# -*- coding: utf-8 -*-
"""Benchmark of all implemented algorithms
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from time import time

# supress warnings for clean output
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pandas import read_excel
#from sklearn.model_selection import train_test_split

from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.knn import KNN
from pyod.models.sod import SOD

#from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score, average_precision_score

def normalizeData(data_og):
    data_max = np.max(data_og, axis=0)
    data_max_v = np.max(data_og)
    data_min = np.min(data_og, axis=0)
    data_min_v = np.min(data_og)
    data_mean = np.mean(data_og, axis=0)
    data_mean_v = np.mean(data_og)
    data_range = data_max - data_min
    data_range_v = data_max_v - data_min_v
    data = (data_og - data_mean) / data_range
    return data, data_mean, data_range, data_mean_v, data_range_v


BASE_DIR = '../../'
# Define data file and read X and y
datasets = ['MED-WTG1-DATA.xlsx',
            'MED-WTG2-DATA.xlsx',
            'MED-WTG3-DATA.xlsx',
            'MED-WTG4-DATA.xlsx',
            'MED-WTG5-DATA.xlsx',
            'MED-WTG6-DATA.xlsx',
            'MED-WTG7-DATA.xlsx',
            'MED-WTG8-DATA.xlsx',
            ]

df_columns = ['Data', '#Samples', '#Dimensions', 'Outlier Perc',
              'LOF', 'COF', 'CBLOF','HBOS', 'KNN', 'AvgKNN',
              'MedKNN', 'SOD']

# define the number of iterations
n_ite = 10
n_classifiers = len(df_columns)-4

# initialize the container for saving the results
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)

for j in range(len(datasets)):

    data_file = BASE_DIR + datasets[j]
    data_name = data_file[len(BASE_DIR):-5]
    data = read_excel(data_file).to_numpy()[:,[2,3,5,7,10,11]]
    data = np.array(data, dtype=np.float32)

    X = data[:,:-1]
    y = data[:,-1]
    outliers_fraction = np.count_nonzero(y) / len(y)
    if outliers_fraction > 0.5:
        outliers_fraction = 0.5
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)

    # construct containers for saving results
    roc_list = [data_name, X.shape[0], X.shape[1], outliers_percentage]
    ap_list = [data_name, X.shape[0], X.shape[1], outliers_percentage]
    prn_list = [data_name, X.shape[0], X.shape[1], outliers_percentage]
    time_list = [data_name, X.shape[0], X.shape[1], outliers_percentage]

    roc_mat = np.zeros([n_ite, n_classifiers])
    ap_mat = np.zeros([n_ite, n_classifiers])
    prn_mat = np.zeros([n_ite, n_classifiers])
    time_mat = np.zeros([n_ite, n_classifiers])

    for i in range(n_ite):
        print("\n... Processing", data_name, '...', 'Iteration', i + 1)
        #random_state = np.random.RandomState(i)

        # 60% data for training and 40% for testing
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random_state)

        # standardizing data for processing
        #X_train_norm, X_test_norm = standardizer(X_train, X_test)
        X_norm = normalizeData(X)
        print(np.shape(X))
        print(np.shape(X_norm))

        classifiers = {
            'Local Outlier Factor (LOF)': LOF(
                contamination=outliers_fraction
            ),
            'Connectivity-Based Outlier Factor (COF)': COF(
                contamination=outliers_fraction
            ),
            'K Nearest Neighbors (KNN)': KNN(
                contamination=outliers_fraction
            ),
            'Average K Nearest Neighbors (AvgKNN)': KNN(
                method='mean',
                contamination=outliers_fraction
            ),
            'Median K Nearest Neighbors (MedKNN)': KNN(
                method='median',
                contamination=outliers_fraction
            ),
            'Subspace Outlier Detection (SOD)': SOD(
                contamination=outliers_fraction
            ) 
        }
        classifiers_indices = {
            'Local Outlier Factor (LOF)': 0,
            'Connectivity-Based Outlier Factor (COF)': 1,
            'K Nearest Neighbors (KNN)': 2,
            'Average K Nearest Neighbors (AvgKNN)': 3,
            'Median K Nearest Neighbors (MedKNN)': 4,
            'Subspace Outlier Detection (SOD)': 5
        }

        for clf_name, clf in classifiers.items():
            t0 = time()
            clf.fit(X_norm)
            scores = clf.decision_function(X_norm)
            t1 = time()
            duration = round(t1 - t0, ndigits=4)

            roc = round(roc_auc_score(y, scores), ndigits=4)
            ap = round(average_precision_score(y, scores), ndigits=4)
            prn = round(precision_n_scores(y, scores), ndigits=4)

            print('{clf_name} ROC:{roc}, AP:{ap}, precision @ rank n:{prn}, '
                  'execution time: {duration}s'.format(
                clf_name=clf_name, roc=roc, ap=ap, prn=prn, duration=duration))

            time_mat[i, classifiers_indices[clf_name]] = duration
            roc_mat[i, classifiers_indices[clf_name]] = roc
            ap_mat[i, classifiers_indices[clf_name]] = ap
            prn_mat[i, classifiers_indices[clf_name]] = prn

    time_list = time_list + np.mean(time_mat, axis=0).tolist()
    temp_df = pd.DataFrame(time_list).transpose()
    temp_df.columns = df_columns
    time_df = pd.concat([time_df, temp_df], axis=0)

    roc_list = roc_list + np.mean(roc_mat, axis=0).tolist()
    temp_df = pd.DataFrame(roc_list).transpose()
    temp_df.columns = df_columns
    roc_df = pd.concat([roc_df, temp_df], axis=0)

    ap_list = ap_list + np.mean(ap_mat, axis=0).tolist()
    temp_df = pd.DataFrame(ap_list).transpose()
    temp_df.columns = df_columns
    ap_df = pd.concat([ap_df, temp_df], axis=0)

    prn_list = prn_list + np.mean(prn_mat, axis=0).tolist()
    temp_df = pd.DataFrame(prn_list).transpose()
    temp_df.columns = df_columns
    prn_df = pd.concat([prn_df, temp_df], axis=0)

    # Save the results for each run
    time_df.to_csv('time.csv', index=False, float_format='%.3f')
    roc_df.to_csv('roc.csv', index=False, float_format='%.3f')
    ap_df.to_csv('ap.csv', index=False, float_format='%.3f')
    prn_df.to_csv('prc.csv', index=False, float_format='%.3f')

writer = pd.ExcelWriter('results_all.xlsx', engine='xlsxwriter')
time_df.to_excel(writer, sheet_name='Execution time', index=False)
roc_df.to_excel(writer, sheet_name='ROC AUC', index=False)
ap_df.to_excel(writer, sheet_name='AP', index=False)
prn_df.to_excel(writer, sheet_name='P@n', index=False)
writer.save()