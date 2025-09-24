import json
import numpy as np
import pandas as pd

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, \
balanced_accuracy_score, ConfusionMatrixDisplay, matthews_corrcoef, roc_auc_score, f1_score, precision_score, \
recall_score, roc_curve
from catboost import CatBoostClassifier, Pool

from utils import *

params_sets = [
    {'norm_sample': True,
     'norm_patient': False,
     'method': 'logfit_4',
     'scale_pat': False},

     {'norm_sample': True,
     'norm_patient': False,
     'method': 'logfit_2',
     'scale_pat': False},

     {'norm_sample': True,
     'norm_patient': False,
     'method': 'mmmms_5',
     'scale_pat': False},

     {'norm_sample': True,
     'norm_patient': False,
     'method': 'mmmms_3',
     'scale_pat': False}
]

dataset = pd.read_csv("dataset_28_08_25.csv")
dataset['datetime'] = pd.to_datetime(dataset['Time'], format='%d/%m/%y %H:%M:%S')

with open('dataset_dict_28_08_25.json') as infile:
    data_save_dict = json.load(infile)

diagnosis_class = {'Z00': 0,
                'E11': 1,
                'K29': 2,
                'K76': 3,
                'B18': 4,
                'C34': 5,
                'N18': 6,
                'J44': 7}

#fold
train_weeks = [[*range(1,9)],
               [*range(4,12)],
               [*range(4,12)],
               [1, *range(5,12)],
               [1, *range(4,10), 11],
               [*range(1,8), 11],
               [1,*range(5,10), 11],
               [*range(1,5), 8, 9, 10]]

test_weeks = [[9, 10, 11],
               [1, 2, 3],
               [1, 2 ,3],
               [2, 3, 4],
               [2, 3, 10],
               [8, 9, 10],
               [2, 3, 4, 10],
               [5, 6, 7, 11]]

diag_train_weeks = {i:j for i, j in zip(diagnosis_class.keys(), train_weeks)}
diag_test_weeks = {i:j for i, j in zip(diagnosis_class.keys(), test_weeks)}

for pset in params_sets:
    rocs = []
    norm_sample = pset['norm_sample']
    #norm_patient = pset['norm_patient']
    method = pset['method']
    scale_pat = pset['scale_pat']
    for i in range(8):
        print(list(diagnosis_class.keys())[i])

        train_ind = dataset[dataset.Week.isin(list(diag_train_weeks.values())[i])].Patient_id
        train_Y = dataset[dataset.Week.isin(list(diag_train_weeks.values())[i])].D_class
        test_ind = dataset[dataset.Week.isin(list(diag_test_weeks.values())[i])].Patient_id
        test_Y = dataset[dataset.Week.isin(list(diag_test_weeks.values())[i])].D_class    

        targets_train = [0 if t == i else 1 for t in train_Y]
        
        targets_test = [0 if t == i else 1 for t in test_Y]

        X_train_split = get_data(data_save_dict, np.tile(train_ind, 1), filter=True, norm_sample=norm_sample, features=FEATURES, delimeter=1.0)
        X_test_split = get_data(data_save_dict, np.tile(test_ind, 1), filter=True, norm_sample=norm_sample, features=FEATURES, delimeter=1.0)

        if scale_pat:
            X_train_split, X_test_split = scale_patient(X_train_split), scale_patient(X_test_split)

        X_train_split, Y_train_split, failed1 = featurize_data(X_train_split, np.tile(train_ind, 1), targets_train, method=method)
        X_test_split, Y_test_split, failed2 = featurize_data(X_test_split, np.tile(test_ind, 1), targets_test, method=method)

        model = CatBoostClassifier(iterations=1500, learning_rate=0.05, loss_function='Logloss', 
                            thread_count=8, max_depth=10, l2_leaf_reg=8, early_stopping_rounds=5)

        train_pool = Pool(X_train_split, Y_train_split)
        eval_pool = Pool(X_test_split, Y_test_split)
        model.fit(train_pool, eval_set=eval_pool, verbose=False)
        predictions = model.predict_proba(X_test_split)[:, 1]
        roc = roc_auc_score(Y_test_split, predictions)
        rocs.append(roc)
        print(roc)
    
    with open('exp_binary_article_catboost_normsamp_always.txt', 'a') as exp_file:
        exp_file.write(' '.join([f'{k}: {v}' for k, v in pset.items()]) + '\n')
        for i, j in zip(rocs, diagnosis_class.keys()):
            exp_file.write(f'{round(i,3)} {j}\n')

    exp_file.close()

