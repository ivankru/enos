import json
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import minmax_scale

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
from sktime.datasets import load_basic_motions

import tensorflow as tf

with open('../datasets/dataset_dict_28_08_25.json') as infile:
    data_save_dict = json.load(infile)

live_channels = set(['R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R2', 'R3', 'R5', 'R8'])

pid2data = {}
for key in data_save_dict:
    row_data = []
    for channel in data_save_dict[key]['sensors'][0]['channels']:
        if channel['id'] in live_channels:
            channel_data = minmax_scale(channel['samples'], axis=0)
            row_data.append(
                pd.Series(channel_data[:357])
            )
    #if len(set([len(j) for j in row_data]))==1 and len(row_data[0])==357:
    pid2data[key] = row_data
len(pid2data)

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

dataset = pd.read_csv("../datasets/dataset_28_08_25.csv")
dataset['datetime'] = pd.to_datetime(dataset['Time'], format='%d/%m/%y %H:%M:%S')
mask = dataset.Patient_id.values!=1113
dataset = dataset.iloc[mask]

n_samples = dataset.Patient_id.shape[0]

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

n_splits = 5

rocs_df = pd.DataFrame(columns = ['Diag'] + [str(i) for i in range(1, n_splits + 1)] + ['mean', 'std'])

for i, cl in enumerate(diagnosis_class.keys()):

        rocs_class = []
        class_name = list(diagnosis_class.keys())[i]
        print(class_name)

        train_val_ind = np.array(dataset[dataset.Week.isin(list(diag_train_weeks.values())[i])].Patient_id)

        test_ind = np.array(dataset[dataset.Week.isin(list(diag_test_weeks.values())[i])].Patient_id)
        test_Y = dataset[dataset.Week.isin(list(diag_test_weeks.values())[i])].D_class
        y_test = np.array(['class1' if j else 'class0' for j in test_Y.values==diagnosis_class[cl]])


        for random_seed in range(n_splits):
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
            y_train = np.array(['class1' if j else 'class0' for j in train_Y.values==diagnosis_class[cl]])

            rows = []
            for i in train_ind:
                i = str(i)
                if i in pid2data:
                    rows.append(pid2data[i])
                else:
                    raise Exception()
            X_train = pd.DataFrame(rows, columns=['dim_'+str(i) for i in range(11)])
            
            rows = []
            for i in test_ind:
                i = str(i)
                if i in pid2data:
                    rows.append(pid2data[i])
                else:
                    raise Exception()
            X_test = pd.DataFrame(rows, columns=['dim_'+str(i) for i in range(11)])
            
            network = InceptionTimeClassifier(n_epochs=70, verbose=False)
            network.fit(X_train, y_train)
            network.score(X_test, y_test)
            
            pred = network.predict_proba(X_test)
            
            gt_y = np.array([int(j[-1]) for j in y_test])
            roc_auc = roc_auc_score(gt_y, pred[:,1])
            rocs_class.append(roc_auc)
            print(cl, roc_auc)

        mean = np.mean(rocs_class)
        std = np.std(rocs_class, ddof = 1)
        rocs_df.loc[len(rocs_df.index)] = [class_name] + [*rocs_class] + [mean, std]
        print(f'{mean} +- {std} for {class_name}')

rocs_df.to_csv('exp_binary_article_incTime_valid_split_weeks.csv', index=False)
    

