import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, \
balanced_accuracy_score, ConfusionMatrixDisplay, matthews_corrcoef, roc_auc_score, f1_score, precision_score, \
recall_score, roc_curve
import xgboost

from utils import *

import warnings
warnings.filterwarnings('ignore')

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

dataset = pd.read_csv("../datasets/dataset_28_08_25.csv")
dataset['datetime'] = pd.to_datetime(dataset['Time'], format='%d/%m/%y %H:%M:%S')

classes = [0, 1, 2, 3, 4, 5, 6, 7]

with open('../datasets/dataset_dict_28_08_25.json') as infile:
    data_save_dict = json.load(infile)

diagnosis_class = {'Z00': 0,
                'E11': 1,
                'K29': 2,
                'K76': 3,
                'B18': 4,
                'C34': 5,
                'N18': 6,
                'J44': 7
                }

#fold
train_weeks = [[*range(1,9)],
               [*range(4,12)],
               [*range(4,12)],
               [1, *range(5,12)],
               [1, *range(4,10), 11],
               [*range(1,8), 11],
               [1,*range(5,10), 11],
               [*range(1,5), 8, 9, 10]
               ]

test_weeks = [[9, 10, 11],
               [1, 2, 3],
               [1, 2 ,3],
               [2, 3, 4],
               [2, 3, 10],
               [8, 9, 10],
               [2, 3, 4, 10],
               [5, 6, 7, 11]
               ]

diag_train_weeks = {i:j for i, j in zip(diagnosis_class.keys(), train_weeks)}
diag_test_weeks = {i:j for i, j in zip(diagnosis_class.keys(), test_weeks)}

n_samples = dataset.Patient_id.shape[0]
print(f'Dataset has {n_samples} patients')

rocs_df = pd.DataFrame(columns = ['Diag', 'Method'] + [str(i) for i in range(1, 17)] + ['mean', 'std'])

for pset in params_sets:
    norm_sample = pset['norm_sample']
    #norm_patient = pset['norm_patient']
    method = pset['method']
    scale_pat = pset['scale_pat']
    print(f'Param set: {pset}')

    for i, cl in tqdm(enumerate(classes)):

        rocs_class = []
        class_name = list(diagnosis_class.keys())[i]
        print(class_name)

        train_val_ind = np.array(dataset[dataset.Week.isin(list(diag_train_weeks.values())[i])].Patient_id)

        test_ind = np.array(dataset[dataset.Week.isin(list(diag_test_weeks.values())[i])].Patient_id)
        test_Y = dataset[dataset.Week.isin(list(diag_test_weeks.values())[i])].D_class
        targets_test = [0 if t == cl else 1 for t in test_Y]
        X_test_split = get_data(data_save_dict, np.tile(test_ind, 1), filter=True, norm_sample=norm_sample, features=FEATURES, delimeter=1.0)
        if scale_pat:
                X_test_split = scale_patient(X_test_split)
        X_test_split, Y_test_split, failed2 = featurize_data(X_test_split, np.tile(test_ind, 1), targets_test, method=method)
        
        for random_seed in range(16):
            np.random.seed(103*random_seed)

            #randomly select 15% of train subset for validation
            n_val = int(train_val_ind.size * 0.15)
            indices = np.random.choice(train_val_ind.size, size=n_val, replace=False)
            val_ind = train_val_ind[indices]
            train_ind = np.delete(train_val_ind, indices, axis=0)

            #print(test_ind[:5], train_ind[:5], val_ind[:5])
            #print(len(test_ind), len(val_ind), len(train_ind))
            assert len(test_ind) + len(val_ind) + len(train_ind) == n_samples

            train_Y = dataset[dataset.Patient_id.isin(list(train_ind))].D_class
            val_Y = dataset[dataset.Patient_id.isin(list(val_ind))].D_class
            targets_train = [0 if t == cl else 1 for t in train_Y]
            targets_val = [0 if t == cl else 1 for t in val_Y]
            X_train_split = get_data(data_save_dict, np.tile(train_ind, 1), filter=True, norm_sample=norm_sample, features=FEATURES, delimeter=1.0)
            X_val_split = get_data(data_save_dict, np.tile(val_ind, 1), filter=True, norm_sample=norm_sample, features=FEATURES, delimeter=1.0)
            
            if scale_pat:
                X_train_split, X_val_split = scale_patient(X_train_split), scale_patient(X_val_split)

            X_train_split, Y_train_split, failed1 = featurize_data(X_train_split, np.tile(train_ind, 1), targets_train, method=method)
            X_val_split, Y_val_split, failed3 = featurize_data(X_val_split, np.tile(test_ind, 1), targets_val, method=method)


            model = xgboost.XGBClassifier(max_depth=8, objective='binary:logistic', eval_metric='logloss',
                        early_stopping_rounds=10, thread_count=8,
                        n_estimators=5000, learning_rate=0.07,
                        reg_lambda=2,
                        random_seed=42, gamma=0.1)

            train_pool = (X_train_split, Y_train_split)
            eval_pool = (X_val_split, Y_val_split)
            model.fit(X_train_split, Y_train_split, eval_set=[eval_pool], verbose=False)
            predictions = model.predict_proba(X_test_split)[:, 1]
            roc = roc_auc_score(Y_test_split, predictions)
            rocs_class.append(roc)
            print(roc)
        
        mean = np.mean(rocs_class)
        std = np.std(rocs_class, ddof = 1)
        rocs_df.loc[len(rocs_df.index)] = [class_name,  method] + [*rocs_class] + [mean, std]
        print(f'{mean} +- {std} for {class_name}, {method}')

rocs_df.to_csv('exp_binary_article_xgboost_valid_split_weeks.csv', index=False)
    
