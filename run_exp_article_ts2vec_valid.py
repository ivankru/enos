# This script is to be luanched from a ts2vec folder, obtained by git clone https://github.com/zhihanyue/ts2vec.git

import sys, os
sys.path.append(os.path.abspath('../'))

from ts2vec import TS2Vec
import datautils
import tasks
from utils import init_dl_program

import json
import numpy as np
import pandas as pd
import torch
import random

import warnings
warnings.filterwarnings('ignore')

import sys
path = os.path.join(os.environ.get('HOME'), 'workdir/enose/')
sys.path.append(path)

from enose.utils_no_typing import * #enose is a folder containing all project code frome anon github
# no typing was needed since ts2vec requires python 3.8 but it seems to work fine with python 3.10 anyway

def seed_everything(seed, workers: bool = False) -> int:

    #print (f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

dataset = pd.read_csv(os.path.join(path, "datasets/dataset_28_08_25.csv"))
dataset['datetime'] = pd.to_datetime(dataset['Time'], format='%d/%m/%y %H:%M:%S')
mask = dataset.Patient_id.values!=1113
dataset = dataset.iloc[mask]
n_samples = dataset.Patient_id.shape[0]

classes = [0, 1, 2, 3, 4, 5, 6, 7]

with open(os.path.join(path, 'datasets/dataset_dict_28_08_25.json')) as infile:
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

norm_sample = True
scale_pat = False

n_splits = 5

seed_everything(369)
device = 'cuda:0'
torch.cuda.set_device(device)
torch.cuda.manual_seed(369)

rocs_df = pd.DataFrame(columns = ['Diag'] + [str(i) for i in range(1, n_splits + 1)] + ['mean', 'std'])

for i, cl in enumerate(classes):

    rocs_class = []
    class_name = list(diagnosis_class.keys())[i]
    print(class_name)

    train_val_ind = np.array(dataset[dataset.Week.isin(list(diag_train_weeks.values())[i])].Patient_id)

    test_ind = np.array(dataset[dataset.Week.isin(list(diag_test_weeks.values())[i])].Patient_id)
    test_Y = dataset[dataset.Week.isin(list(diag_test_weeks.values())[i])].D_class
    targets_test = [0 if t == cl else 1 for t in test_Y]
    X_test_split = get_data(data_save_dict, np.tile(test_ind, 1), filter=True, norm_sample=norm_sample, features=FEATURES, delimeter=1.0)
    X_test_split = X_test_split.transpose(0, 2, 1)
    if scale_pat:
            X_test_split = scale_patient(X_test_split)
    
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
        val_Y = dataset[dataset.Patient_id.isin(list(val_ind))].D_class
        targets_train = [0 if t == cl else 1 for t in train_Y]
        targets_val = [0 if t == cl else 1 for t in val_Y]
        X_train_split = get_data(data_save_dict, np.tile(train_ind, 1), filter=True, norm_sample=norm_sample, features=FEATURES, delimeter=1.0)
        X_val_split = get_data(data_save_dict, np.tile(val_ind, 1), filter=True, norm_sample=norm_sample, features=FEATURES, delimeter=1.0)
        
        if scale_pat:
            X_train_split, X_val_split = scale_patient(X_train_split), scale_patient(X_val_split)

        X_train_split = X_train_split.transpose(0, 2, 1)
        Y_train, Y_test = np.array(targets_train), np.array(targets_test)

        print(X_train_split.shape, Y_train.shape, X_test_split.shape, Y_test.shape)

        model = TS2Vec(
        input_dims=8,
        device=device,
        output_dims=320,
        batch_size=64,
        lr=0.001
        )

        loss_log = model.fit(
            X_train_split,
            n_epochs=40,
            verbose=True,
        )

        out, eval_res = tasks.eval_classification(model, X_train_split, Y_train, X_test_split, Y_test, eval_protocol='linear')
        print(eval_res)

        roc = eval_res['roc_auc']
        rocs_class.append(roc)

    mean = np.mean(rocs_class)
    std = np.std(rocs_class, ddof = 1)
    rocs_df.loc[len(rocs_df.index)] = [class_name] + [*rocs_class] + [mean, std]
    print(f'{mean} +- {std} for {class_name}')

rocs_df.to_csv('exp_binary_article_ts2vec_valid_split_weeks.csv', index=False)
