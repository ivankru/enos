import os
import random
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import PolynomialFeatures
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt

import pywt
from scipy.signal import medfilt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

import json

from conv_models import *

def seed_everything(seed, workers: bool = False) -> int:

    #print (f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    print(f'seed is set to {seed}')

if not os.path.exists("fold1"):
    os.mkdir("fold1")
if not os.path.exists("folds"):
    os.mkdir("folds")

df = pd.read_csv(os.path.join(os.environ.get('HOME'), 'workdir/enose/datasets/dataset_28_08_25.csv'))
inverted_diagnosis_class = {0: 'Z00',
 1: 'E11',
 2: 'K29',
 3: 'K76',
 4: 'B18',
 5: 'C34',
 6: 'N18',
 7: 'J44'}
df.Diagnosis = df.apply(lambda row: inverted_diagnosis_class[row['D_class']], axis = 1)
pid2icd = {}
for i, v in zip(df.Patient_id.values, df.Diagnosis.values):
    pid2icd[i] = v.split(' ')[-1]

pid2datetime = {}
for i, v in zip(df.Patient_id.values, df.datetime.values):
    # Convert to datetime object
    dt_object = datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
    # Convert to timestamp (seconds since epoch)
    timestamp = dt_object.timestamp()
    pid2datetime[i] = int(timestamp)

data = pd.read_json(os.path.join(os.environ.get('HOME'), 'workdir/enose/datasets/dataset_dict_28_08_25.json'))

live_channels = []
for i, (patient_id, value) in enumerate(zip(data.columns, data.values[-1])):    
    valid_channels = set()
    for channel_dict in value[0]['channels']:
        cid = channel_dict['id']
        sig_unique = len(set(channel_dict['samples']))
        if sig_unique!=1:
            valid_channels.add(cid)
    live_channels.append(valid_channels)
    #print ('#', i, 'id', patient_id, valid_channels)
    #break
union = set.intersection(*live_channels)

datetime_series = pd.to_datetime(data.values[4])
week_numbers = datetime_series.isocalendar().week.to_numpy()
week_numbers = week_numbers-np.min(week_numbers)+1

pid2meta = {}
for patient_id, bd, gender, date, week_number in zip(data.columns, data.values[1], data.values[2], datetime_series, week_numbers):
    #print (patient_id, bd, type(date), week_number)
    parts = bd.split('/')
    yearpart = parts[-1]
    if int(yearpart)<20:
        bd_new = '/'.join(parts[:2]+['20'+parts[2]])
    else:
        bd_new = '/'.join(parts[:2]+['19'+parts[2]])
    date_from_string = pd.to_datetime(bd_new, format='%m/%d/%Y')
    
    # Make timestamp date timezone-naive (remove timezone info)
    
    timestamp_date_naive = date.tz_localize(None)

    # Calculate age in years
    age_in_years = int((timestamp_date_naive - date_from_string).days / 365.25)
    
    pid2meta[patient_id] = (gender, age_in_years, week_number)

mm_scale = True

pid2data = {}
for i, (patient_id, value) in enumerate(zip(data.columns, data.values[-1])):
    pid2data[patient_id] = []
    for channel_dict in value[0]['channels']:
        cid = channel_dict['id']
        if cid in union:
            #print (cid)
            series = channel_dict['samples']#[8:180]
            
            coeffs = pywt.wavedec(series, 'db4', level=4)
            coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
            smoothed_data = pywt.waverec(coeffs, 'db4')

            #smoothed_data = medfilt(series, kernel_size=5)
            
            if mm_scale:
                result_data = minmax_scale(smoothed_data, axis=0)
            else: 
                result_data = smoothed_data

            
            pid2data[patient_id].append(result_data)
    pid2data[patient_id] = np.array(pid2data[patient_id])
    #break

    # polynomial features

original_data = pid2data[18]
poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
poly.fit_transform(original_data.T).T  # (11 + C(11,2) + 11 = 77, 358)

seed_everything(369)

# generate 256 random combinations
n_new_channels = 256
n_base_channels = 374 #172 #old 374

# random weights
weights = np.random.uniform(-13, 13, (n_new_channels, n_base_channels))

pid2data_ext = {}
for pid in pid2data:
    original_data = pid2data[pid]
    poly_features = poly.transform(original_data.T).T  # (11 + C(11,2) + 11 = 77, 358)

    extended_features = np.vstack([original_data, poly_features])
    
    # Make new channels
    new_channels = np.dot(weights, extended_features)  # (256, 358)
    
    #nonlinear_expanded = apply_nonlinearities(original_data)

    # 4. combine with orignial data if needed 
    final_data = np.vstack(
        [original_data, new_channels]
    )  # (11 + 256 = 267, 358)

    # 5. Normalize (optional)
    final_data = (final_data - np.mean(final_data, axis=1, keepdims=True)) / np.std(final_data, axis=1, keepdims=True)
    
    pid2data_ext[pid] = final_data

n_channels = final_data.shape[0]

code2id = {v: i for i, v in enumerate(sorted(set(pid2icd.values())))}
id2code = {code2id[key]:key for key in code2id}

diagnosis_class = {value: key for key, value in inverted_diagnosis_class.items()}
pid_ts_values = sorted(pid2datetime.items(), key=lambda item: item[1])
indices = np.array([pid for pid, ts in pid_ts_values])
codes_by_time = np.array([pid2icd[pid] for pid, ts in pid_ts_values])

fold1 = {
    #'code': (train_weeks, train_pos, val_weeks, val_pos)
    'K29': (set([4,5,6,7,8,9,10,11]), 99, set([1,2,3]), 39),
    'B18': (set([1,4,5,6,7,8,9,11]), 109, set([2,3,10]), 29),
    'C34': (set([1,2,3,4,5,6,7,11]), 64, set([8,9,10]), 36),
    'Z00': (set([1,2,3,4,5,6,7,8]), 133, set([9,10,11]), 34),
    'K76': (set([1,5,6,7,8,9,10,11]), 96, set([2,3,4]), 32),
    'E11': (set([4,5,6,7,8,9,10,11]), 91, set([1,2,3]), 37),
    'J44': (set([1,2,3,4,8,9,10]), 80, set([5,6,7,11]), 20),
    'N18': (set([1,5,6,7,8,9,11]), 101, set([2,3,4,10]), 27)
}

for dir_name, fld in [('fold1', fold1)]:
    for code in code2id:
        trainweeks = fld[code][0]
        valweeks = fld[code][2]
        train_ids, val_ids = [], []
        for pid in pid2data_ext.keys():
            week = pid2meta[pid][2]
            if week in trainweeks:
                train_ids.append(pid)
            else:
                val_ids.append(pid)

        with open(dir_name+'/'+code+'.val.fold.0.txt', 'w') as f:
            s = ' '.join([str(i) for i in val_ids])
            f.write(s)

print('Constructing dataset')
X, y, ids = [], [], []
for pid in pid2data_ext:
    h,w = pid2data_ext[pid].shape
    if h==n_channels and w>=358 and pid2icd[pid] in code2id:
        X.append( pid2data_ext[pid][:,:358] )
        code = pid2icd[pid]
        y.append(code2id[code])
        ids.append(pid)

print(len(X), len(y), len(ids))
X = np.array(X).astype(np.float32)
y = np.array(y)
ids = np.array(ids)
print(X.shape, y.shape, ids.shape)

#train CNN

seed_everything(369)
device = 'cuda:1'
torch.cuda.set_device(device)
torch.cuda.manual_seed(369)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

rocs_df = pd.DataFrame(columns = ['Diag'] + [str(i) for i in range(1, 17)] + ['mean', 'std'])

for splt in ['fold1']:
    print (splt)
    for code in code2id:
        scores = []
        for fold in range(1):
            with open(splt+'/'+code+'.val.fold.'+str(fold)+'.txt') as f:
                ids_cv = np.array([int(i) for i in f.readline().split(' ')])

            mask_cv = np.isin(ids, ids_cv)
            X_cv, y_cv = X[mask_cv], y[mask_cv]
            y_cv_bin = y_cv==code2id[code]

            ids_train_val = ids[~mask_cv]

            dataset_cv = CustomDataset(X_cv, y_cv_bin, transform=transform_cnn)
            cv_loader = DataLoader(dataset_cv, batch_size=16, shuffle=False, num_workers=8)

            for random_seed in range(16):
                np.random.seed(103*random_seed)

                #randomly select 15% of train subset for validation
                n_val = int(ids_train_val.size * 0.15)
                indices = np.random.choice(ids_train_val.size, size=n_val, replace=False)
                val_ind = ids_train_val[indices]
                train_ind = np.delete(ids_train_val, indices, axis=0)

                mask_val = np.isin(ids, val_ind)
                mask_train = np.isin(ids, train_ind)

                X_train, X_val = X[mask_train], X[mask_val]
                y_train, y_val = y[mask_train], y[mask_val]

                y_train_bin, y_val_bin = y_train==code2id[code], y_val==code2id[code]

                dataset_train = CustomDataset(X_train, y_train_bin, transform=transform_cnn)
                dataset_val = CustomDataset(X_val, y_val_bin, transform=transform_cnn)
                
                train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=8)
                val_loader = DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=8)

                # init
                model = HeightWiseCNN(num_classes=2,n_channels=n_channels).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=5e-5)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

                num_epochs = 40
                best_val_auc = 0.0
                best_path = "heightwise_cnn_best.pth"

                for epoch in range(num_epochs):
                    # ---- Train ----
                    model.train()
                    tr_loss_sum, tr_count = 0.0, 0
                    outputs, targets = [], []
                    for batch_idx, (data, target) in enumerate(train_loader):
                        data, target = data.to(device), target.to(device, dtype=torch.long)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                        bs = data.size(0)
                        tr_loss_sum += loss.item() * bs

                        outputs.append(output)
                        targets.append(target)
                        tr_count += bs



                    tr_loss = tr_loss_sum / tr_count
                    outputs, targets = torch.cat(outputs), torch.cat(targets)
                    tr_auc = roc_auc(outputs, targets)

                    # ---- Val ----
                    model.eval()
                    val_loss_sum, val_count = 0.0, 0
                    outputs, targets = [], []
                    with torch.no_grad():
                        for data, target in val_loader:
                            data, target = data.to(device), target.to(device, dtype=torch.long)
                            output = model(data)
                            loss = criterion(output, target)
                            bs = data.size(0)
                            val_loss_sum += loss.item() * bs
                            outputs.append(output)
                            targets.append(target)
                            #probabilities = F.softmax(output, dim=1)
                            val_count += bs

                    val_loss = val_loss_sum / val_count

                    outputs, targets = torch.cat(outputs), torch.cat(targets)
                    val_auc = roc_auc(outputs, targets)

                    scheduler.step()

                    #print(f"Epoch {epoch:02d}: "
                    #     f"train_loss={tr_loss:.4f} train_auc={tr_auc:.4f} | "
                    #    f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} | "
                    #   f"lr={optimizer.param_groups[0]['lr']:.2e}")

                    # best val checkpoint
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        torch.save({
                            "model_state": model.state_dict(),
                            "val_auc": best_val_auc,
                            "epoch": epoch,
                        }, best_path)
                        #print (f"\tNew best checkpoint: val_auc={best_val_auc:.4f}")

                        with open('folds/'+code+'.fold.'+str(fold)+'.txt', 'w') as f:
                            s = ' '.join([str(i) for i in np.array(targets.cpu().numpy())])
                            f.write(s)
                            f.write('\n')
                            s = ' '.join([str(i) for i in np.array((1/(1+torch.exp(-outputs))[:,1]).cpu().numpy())])
                            f.write(s)
                # ---- Test ----
                model.load_state_dict(torch.load(best_path)['model_state'])
                model.eval()
                outputs, targets = [], []
                with torch.no_grad():
                    for data, target in cv_loader:
                        data, target = data.to(device), target.to(device, dtype=torch.long)
                        output = model(data)
                        outputs.append(output)
                        targets.append(target)

                outputs, targets = torch.cat(outputs), torch.cat(targets)
                cv_auc = roc_auc(outputs, targets)
                scores.append(cv_auc)
                print(cv_auc)
        print(scores)
        print ('code {} ({}/{}) {}+-{}'.format(code, np.sum(y_cv_bin), y_cv_bin.shape[0], np.mean(scores), np.std(scores, ddof = 1)))

        mean = np.mean(scores)
        std = np.std(scores, ddof = 1)
        rocs_df.loc[len(rocs_df.index)] = [code] + [*scores] + [mean, std]

rocs_df.to_csv('exp_binary_article_cnn_valid_split_weeks.csv', index=False)
            