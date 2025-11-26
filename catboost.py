import json
import numpy as np
import pandas as pd

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, \
balanced_accuracy_score, ConfusionMatrixDisplay, matthews_corrcoef, roc_auc_score, \
f1_score, precision_score, \
recall_score, roc_curve
from catboost import CatBoostClassifier, Pool
from dateutil.relativedelta import relativedelta
from utils.utils_ivp import *
import warnings
warnings.filterwarnings("ignore")

def get_data_gender_age(id_class_dict_gender: dict, id_class_dict_age,
             ids: list, start = START, end = END, 
             sample_rate = SAMPLE_RATE, features = FEATURES, 
             trim = True, 
             filter = True,
             norm_sample = False,
             delimeter:float = 1.0):

    features_list = []
    for patient_id in ids:
        age = id_class_dict_age[patient_id]
        gender = 1 if id_class_dict_gender[patient_id] == 'Мужской' else 0
        features_list.append([gender, age])
    features_list = np.array(features_list)
    return features_list


def featurize_data(data, x, Y: np.ndarray, method = 'logfit_4', aug_factor: int = 1):

    n_samples, n_features, n_points = data.shape
    failed = []
    result_x = []
    result_Y = []
    result_Y_all = []


    assert n_samples == len(Y)

    if method.startswith('logfit'):
        num_par = int(method.split('_')[1])
        
        for s in range(n_samples):
            result = np.zeros((n_features, num_par))
            result_Y_all.append(Y[s])
            try:
                for f in range(n_features):
                    t1 = np.arange(n_points)
                    _, _, line1_par, _ = fit_logistic(data[s, f], t1, num_par)
                    result[f] = line1_par
                
                result_x.append(result.reshape(n_features * num_par))
                result_Y.append(Y[s])
            except RuntimeError:
                ind = s
                while not ind < len(x):
                    ind -= aug_factor*len(x)
                #print(f'Failed to feturize {x[ind]}, {ind}, {s}')
                failed.append(x[ind])
                continue

        return np.array(result_x), np.array(result_Y), failed, np.array(result_Y_all)
    
    elif method.startswith('mmmms'): # min max mean median std or mean median std
        num_par = int(method.split('_')[1])

        for s in range(n_samples):
            result = np.zeros((n_features, num_par))
            
            for f in range(n_features):
                min_v = data[s, f].min()
                max_v = data[s, f].max()
                mean_v = data[s, f].mean()
                median_v = np.median(data[s, f])
                std_v = data[s, f].std(ddof = 1)
                if num_par == 3:
                    line1_par = [mean_v, median_v, std_v]
                elif num_par == 5:
                    line1_par = [min_v, max_v, mean_v, median_v, std_v]
                else:
                    return
                
                result[f] = line1_par
            
            result_x.append(result.reshape(n_features * num_par))
            result_Y.append(Y[s])

        return np.array(result_x), np.array(result_Y), failed

    else:       
        return



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

dataset = pd.read_csv("data/dataset_28_08_25.csv")
dataset['datetime'] = pd.to_datetime(dataset['Time'], format='%d/%m/%y %H:%M:%S')

dataset_good = dataset[dataset['Comment'].isna()].reset_index(drop=True)
dataset_sorted_good = dataset_good.sort_values(by='datetime')
id_for_classes = dataset_sorted_good['Patient_id']
gender_list = dataset_sorted_good['Patient_gender']
birth_data = dataset_sorted_good['Patient_bd']
s_datetime = pd.to_datetime(birth_data, format='%m/%d/%y')
def subtract_100_years(dt):
    return dt - relativedelta(years=100)

# convert birth date into full years of life
target_date = pd.to_datetime('2025-08-01')
# timedelta
# years < 70 are tranlated as 20**
s_datetime[s_datetime > target_date] = s_datetime[s_datetime > target_date].apply(subtract_100_years)
time_difference = target_date - s_datetime
# days into years approximatly
years_approx = time_difference.dt.days / 365.25

id_class_dict_gender = dict(zip(id_for_classes, gender_list))
indices_men = [key for key, value in id_class_dict_gender.items() if value == 'Мужской']
indices_woman = [key for key, value in id_class_dict_gender.items() if value == 'Женский']
id_class_dict_age = dict(zip(id_for_classes, years_approx))
#55 - median age
indices_young = [key for key, value in id_class_dict_age.items() if value <= 55]
indices_eldely = [key for key, value in id_class_dict_age.items() if value > 55]

with open('data/dataset_dict_28_08_25.json') as infile:
    data_save_dict = json.load(infile)

diagnosis_class = {'Z00': 0,
                'E11': 1,
                'K29': 2,
                'K76': 3,
                'B18': 4,
                'C34': 5,
                'N18': 6,
                'J44': 7}

#fold 1
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

#fold 2

# test_weeks = [[1, 3],
#                [4, 6, 8],
#                [4],
#                [9, 10],
#                [5],
#                [10],
#                [6, 9, 10],
#                [8, 10]]

# train_weeks = [[2, *range(4,12)],
#                [*range(1,4), 5, 7, *range(9, 12)],
#                [*range(1,4), *range(5, 14)],
#                [*range(1,9), 11],
#                [*range(1,5), *range(6,12)],
#                [*range(1,10), 11],
#                [*range(1,6), 7, 11],
#                [*range(1,8), 9, 11]]


diag_train_weeks = {i:j for i, j in zip(diagnosis_class.keys(), train_weeks)}
diag_test_weeks = {i:j for i, j in zip(diagnosis_class.keys(), test_weeks)}

for pset in params_sets[0:1]:
    roc_mean_list = []
    roc_std_list = []
    norm_sample = pset['norm_sample']
    #norm_patient = pset['norm_patient']
    method = pset['method']
    scale_pat = pset['scale_pat']
    for i in range(8):
        print(list(diagnosis_class.keys())[i])

        roc_list = []
        for random_seed in range(16):
            np.random.seed(103*random_seed)

            train_ind = dataset[dataset.Week.isin(list(diag_train_weeks.values())[i])].Patient_id
            train_Y = dataset[dataset.Week.isin(list(diag_train_weeks.values())[i])].D_class
            test_ind = dataset[dataset.Week.isin(list(diag_test_weeks.values())[i])].Patient_id
            test_Y = dataset[dataset.Week.isin(list(diag_test_weeks.values())[i])].D_class 

            # use if you need to consider only specific categories - gender
            # indices_of_interest = set(indices_eldely)
            # train_ind_old = train_ind
            # test_ind_old = test_ind
            # train_ind = [index for index, value in zip(train_ind_old, train_Y) if index in indices_of_interest]   
            # train_Y = [value for index, value in zip(train_ind_old, train_Y) if index in indices_of_interest]
            # test_ind = [index for index, value in zip(test_ind_old, test_Y) if index in indices_of_interest]   
            # test_Y = [value for index, value in zip(test_ind_old, test_Y) if index in indices_of_interest]      

            targets_train = [0 if t == i else 1 for t in train_Y]
            targets_test = [0 if t == i else 1 for t in test_Y]

            X_train_split_old = get_data(data_save_dict, np.tile(train_ind, 1), filter=True, norm_sample=norm_sample, features=FEATURES, delimeter=1.0)
            X_test_split_old = get_data(data_save_dict, np.tile(test_ind, 1), filter=True, norm_sample=norm_sample, features=FEATURES, delimeter=1.0)
            X_train_split = get_data_gender_age(id_class_dict_gender, id_class_dict_age, np.tile(train_ind, 1))
            X_test_split = get_data_gender_age(id_class_dict_gender, id_class_dict_age, np.tile(test_ind, 1))

            #to select gender or age group
            # targets_train = [0 if t[0] == 0 else 1 for t in X_train_split]
            # targets_test = [0 if t[0] == 0 else 1 for t in X_test_split]

            if scale_pat:
                X_train_split, X_test_split = scale_patient(X_train_split), scale_patient(X_test_split)

            #randomly select 15% of train subset for validation
            n_val = int(X_train_split.shape[0] * 0.15)
            indices = np.random.choice(X_train_split.shape[0], size=n_val, replace=False)
            X_val_split = X_train_split[indices,:]
            #remove validation data from train
            X_train_split = np.delete(X_train_split, indices, axis=0)
            targets_val = np.array(targets_train)[indices]
            targets_train = np.delete(np.array(targets_train), indices, axis=0)
            X_val_split_old = np.array(X_train_split_old)[indices,:]
            #remove validation data from train
            X_train_split_old = np.delete(np.array(X_train_split_old), indices, axis=0)

            # _, _, failed1, Y_train_split = featurize_data(X_train_split_old, np.tile(train_ind, 1), targets_train, method=method)
            # _, _, failed2, Y_test_split = featurize_data(X_test_split_old, np.tile(test_ind, 1), targets_test, method=method)
            X_train_split, Y_train_split, failed1, _ = featurize_data(X_train_split_old, np.tile(train_ind, 1), targets_train, method=method)
            X_test_split, Y_test_split, failed2, _ = featurize_data(X_test_split_old, np.tile(test_ind, 1), targets_test, method=method)

            model = CatBoostClassifier(iterations=1500, learning_rate=0.05, loss_function='Logloss', 
                                thread_count=8, max_depth=10, l2_leaf_reg=8, early_stopping_rounds=5)

            train_pool = Pool(X_train_split, Y_train_split)
            eval_pool = Pool(X_test_split, Y_test_split)
            model.fit(train_pool, eval_set=eval_pool, verbose=False)
            predictions = model.predict_proba(X_test_split)[:, 1]
            roc = roc_auc_score(Y_test_split, predictions)
            h = 0.85 #have to be find individually
            predictions_discrete = predictions > h
            specifity = recall_score(Y_test_split, predictions_discrete)
            sensitivity = recall_score(1-Y_test_split, 1-predictions_discrete)
            precison = precision_score(1-Y_test_split, 1-predictions_discrete)
            recall_macro = balanced_accuracy_score(Y_test_split, predictions_discrete)
            #roc = sensitivity
            print(f"{random_seed}. ROC AUC:{roc:.3f}")
            roc_list.append(roc)
        roc_auc_mean = np.array(roc_list).mean()
        roc_auc_std = np.array(roc_list).std()    
        roc_mean_list.append(roc_auc_mean)
        roc_std_list.append(roc_auc_std)

    print("Statistics:")
    for i, j, k in zip(roc_mean_list, roc_std_list, diagnosis_class.keys()):
        print(f'ROC AUC:{round(i,3)} ± {round(j,3)} {k}\n')

    #output results
    with open('data/exp_binary_article_catboost_normsamp_always.txt', 'a') as exp_file:
        exp_file.write(' '.join([f'{k}: {v}' for k, v in pset.items()]) + '\n')
        for i, j, k in zip(roc_mean_list, roc_std_list, diagnosis_class.keys()):
            exp_file.write(f'ROC AUC:{round(i,3)} ± {round(j,3)} {k}\n')

    exp_file.close()