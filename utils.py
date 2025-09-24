import json
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from io import StringIO
from collections import Counter
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, precision_recall_curve
from scipy.ndimage import median_filter
from sklearn.utils import shuffle

PARAMS_SET = [
    {'split': 'strat',
     'norm_sample': False,
     'norm_patient': False,
     'method': 'logfit_4'},

     {'split': 'random',
     'norm_sample': False,
     'norm_patient': False,
     'method': 'logfit_4'},

     {'split': 'strat',
     'norm_sample': False,
     'norm_patient': False,
     'method': 'logfit_3'},

     {'split': 'random',
     'norm_sample': False,
     'norm_patient': False,
     'method': 'logfit_3'},

    {'split': 'strat',
     'norm_sample': False,
     'norm_patient': False,
     'method': 'logfit_2'},

     {'split': 'random',
     'norm_sample': False,
     'norm_patient': False,
     'method': 'logfit_2'},

     {'split': 'strat',
     'norm_sample': True,
     'norm_patient': False,
     'method': 'logfit_2'},

     {'split': 'random',
     'norm_sample': True,
     'norm_patient': False,
     'method': 'logfit_2'},

     {'split': 'strat',
     'norm_sample': False,
     'norm_patient': True,
     'method': 'logfit_2'},

     {'split': 'random',
     'norm_sample': False,
     'norm_patient': True,
     'method': 'logfit_2'},

     {'split': 'strat',
     'norm_sample': False,
     'norm_patient': True,
     'method': 'logfit_3'},

     {'split': 'random',
     'norm_sample': False,
     'norm_patient': True,
     'method': 'logfit_3'},
]

PARAMS_SET_SMALL = [
    {'split': 'strat',
     'norm_sample': True,
     'norm_patient': False,
     'method': 'logfit_2'},

     {'split': 'random',
     'norm_sample': True,
     'norm_patient': False,
     'method': 'logfit_2'},
]

DATES = ['06-18', '06-19', '06-20', 
         '06-23', '06-24', '06-25', '06-26', '06-27', 
         '06-30', '07-01', '07-02', '07-03', '07-04', 
         '07-07', '07-08', '07-09', '07-10', '07-11',
         '07-14', '07-15', '07-16', '07-17', '07-18',
         '07-21', '07-22', '07-23', '07-24', '07-25',
         '07-28', '07-29', '07-30', '07-31', '08-01',
         '08-04', '08-05', '08-06', '08-07', '08-08',
         '08-11', '08-12', '08-13', '08-14', '08-15',
         '08-18', '08-19', '08-20', '08-21', '08-22',
         '08-25', '08-26', '08-27', '08-28',
         ]

FEATURES_ALL = [
    'R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','R11','R12','R13','R14','R15','R16','R17',
    #'Humidity','Temperature'
]

FEATURES = [
    #'R1',
    #'R2',
    #'R3',
    #'R4',
    #'R5',
    #'R6',
    #'R7',
    'R8',
    #'R9',
    #'R10',
    'R11',
    'R12',
    'R13',
    'R14',
    'R15',
    'R16',
    'R17',
]

START = 20
END = 450
SAMPLE_RATE = 0.4
NUM_FOLDS = 5

PATH = ''

def defaults():
    return {'dates': DATES,
            'features': FEATURES,
            'features_all': FEATURES_ALL,
            'start': START,
            'end': END,
            'sample_rate': SAMPLE_RATE,
            'num_folds': NUM_FOLDS,
            'params_set': PARAMS_SET,
            'params_set_small': PARAMS_SET_SMALL}

def precision_at_specific_recall(y_true, y_pred_proba, target_recall=0.8, tolerance=0.01, pos_class = 0):

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba, pos_label=pos_class)
    
    recall_diffs = np.abs(recall - target_recall)
    valid_indices = np.where(recall_diffs <= tolerance)[0]
    
    if len(valid_indices) == 0:
        closest_idx = np.argmin(recall_diffs)
        return precision[closest_idx], recall[closest_idx]
    
    best_idx = valid_indices[np.argmax(precision[valid_indices])]
    return precision[best_idx], recall[best_idx]

def get_patient_num(file: str) -> tuple[str,bool]:
    with open(file) as infile:
        data = json.load(infile)
        card = StringIO(data['medicalCard'])
        df = pd.read_csv(card, sep='\t')
        num = df['pid'][0]
        if str(num)[-1] in ["'", "`"]:
            num = num[:-1]
        try:
            num = int(num)
            if num < 1300:
                return num, True
            else:
                return str(num), False
        except ValueError:
            num = df['pid'][0]
            return num, False
        
def load_data(dates: list[str] = DATES, path = PATH) -> list[tuple]:
    files = []
    date_search = dates
    exp_data = os.listdir(path)
    exp_data = [file for file in exp_data if file.endswith('.json')]

    for file in exp_data:
        num, valid = get_patient_num(os.path.join(path, file))
        time = ':'.join(file.split('T')[1].split(':')[0:2])
        date = '-'.join(file.split('T')[0].split('2025-')[1].split('-'))
        if date not in date_search or not valid:
            continue
        else:
            files.append((os.path.join(path, file), num, date, time))
    
    return files

def get_data_json(file: str) -> dict:
    sensors = []
    with open(file[0]) as file_data:
        patient_data = json.load(file_data)

    return patient_data

"""def get_data(file, start = START, end = END, sample_rate = SAMPLE_RATE, features = FEATURES, trim = True):
    sensors = []
    with open(file[0]) as file_data:
        patient_data = json.load(file_data)
        for i in features:
            feature = int(i[1:])
            feat_data = patient_data['sensors'][0]['channels'][feature - 1]['samples']
            if trim:
                sensors.append(feat_data[int(start * sample_rate - 1):int(end * sample_rate - 1)])
            else:
                sensors.append(feat_data)
    
    return np.vstack(sensors)"""

def correct_data(files: list[tuple], corrections: list[tuple]) -> list[tuple]:
    corrected = []
    for file in files:
        found = False
        for i, corr in enumerate(corrections):
            if file[1] == corr[0] and file[2] == corr[1] and file[3] == corr[2]:
                found = True
                print(corr)
                print(file)
                if corr[3] == 'del':
                    pass
                elif corr[3] == 'change':
                    new = (file[0], corr[4], file[2], file[3])
                    corrected.append(new)
                    print(new)
                else:
                    raise NotImplementedError(corr[3])
        if found:
            pass
        else:
            corrected.append(file)
    return corrected

def _logistic_time(t, R0, Rmax, t50, k):
    return R0 + (Rmax - R0) / (1 + (t50 / t)**k)

def fit_logistic(y, t, num_par = 4):
    p0_logistic = [min(y), max(y), np.median(t), 1]  # [R0, Rmax, t50, k]
    params_logistic, _ = curve_fit(_logistic_time, t, y, p0=p0_logistic, maxfev=10000)
    R0, Rmax, t50, k = params_logistic

    t_fit = np.linspace(min(t), max(t))
    R_logistic_fit = _logistic_time(t_fit, R0, Rmax, t50, k)
    r2_logistic = r2_score(y, _logistic_time(t, R0, Rmax, t50, k))
    #rmse_logistic = np.sqrt(mean_squared_error(y, logistic_time(t, R0, Rmax, t50, k)))

    #params_logistic = [str(np.round(i, 1)) for i in params_logistic]
    if num_par == 4:
        return R_logistic_fit, r2_logistic, params_logistic, t_fit
    elif num_par == 3:
        params = [params_logistic[1] - params_logistic[0], params_logistic[2], params_logistic[3]]
        return R_logistic_fit, r2_logistic, params, t_fit
    elif num_par == 2:
        return R_logistic_fit, r2_logistic, params_logistic[2:], t_fit

def get_data(dataset_json: dict, ids: list, start = START, end = END, 
             sample_rate = SAMPLE_RATE, features = FEATURES, 
             trim = True, 
             filter = True,
             norm_sample = False,
             delimeter:float = 1.0):
    
    dataset = []
    for patient_id in ids:
        sensors = []
        patient_data = dataset_json[str(patient_id)]
        for i in features:
            feature = int(i[1:])
            feat_data = patient_data['sensors'][0]['channels'][feature - 1]['samples']
            if trim:
                trimmed_data = feat_data[int(start * sample_rate + 2):int((end * sample_rate + 2)/delimeter)]
            else:
                trimmed_data = feat_data

            if filter:
                filtered_data = median_filter(trimmed_data, 15)
            else:
                filtered_data = trimmed_data

            if norm_sample:
                norm_data = normalize_sample(filtered_data)
            else:
                norm_data = filtered_data
            sensors.append(norm_data)

        dataset.append(sensors)
    return np.array(dataset)

def augmentSample(data, method = 'const_percent', seed = 42):

    np.random.seed(seed)
    n_samples, n_features, n_points = data.shape
    
    if method.startswith('const_percent'):

        result = np.zeros((n_samples, n_features, n_points))

        for s in range(n_samples):
            for f in range(n_features):
                aug = data[s, f] * np.random.randint(90, 111) / 100
                median = np.median(data[s, f])
                add = np.random.uniform(-0.005, 0.005, n_points) * median
                aug = aug + add
                result[s, f] = aug

    return result

def augmentData(data: np.ndarray, Y, method: str = 'average_sum', seed = 42):

    if isinstance(Y, pd.Series):
        Y = Y.values
    elif isinstance(Y, np.ndarray):
        pass
    else:
        raise NotImplementedError('Y must be either pandas series or numpy array')

    #np.random.seed(seed)
    n_samples, n_features, n_points = data.shape
    assert n_samples == len(Y)

    if method == 'average_sum':

        counter = Counter(Y).items()
        class_counts = np.array([i[1] for i in counter])
        classes = np.array([i[0] for i in counter])
        n_new = np.sum(2*class_counts - 3)
        result_x = np.zeros((n_new, n_features, n_points))
        result_Y = np.zeros((n_new))

        point = 0
        # Iterate classes
        for cl in classes:
            idxs = shuffle(np.where(Y == cl)[0], random_state=42)
            n_local = len(idxs)
            #print(n_local)

            # Iterate class items, retain two pairs for each element (i, i+1) (i, i+2)
            for i in range(n_local):
                for j in range(i + 1, min(i + 3, n_local)):
                    
                    #print(i, j, idxs[i], idxs[j], cl)
                    # Iterate features
                    for ft in range(n_features):
                        av_sum = (data[idxs[i], ft] + data[idxs[j], ft]) / 2
                        result_x[point, ft] = av_sum
                    result_Y[point] = cl
                    point += 1 

    return result_x, result_Y

def scale_patient(data, Scaler = StandardScaler()):

    data_scaled = data.copy()
    n_samples, n_features, n_points = data.shape
    for i in range(n_samples):
        data_scaled[i] = Scaler.fit_transform(data[i].reshape(n_features*n_points, -1)).reshape(n_features, n_points)
    
    return data_scaled

def scale_batch(scaler, train, test, val=None): #old scale_data

    n_samples, n_features, n_points = train.shape
    train_scaled = scaler.fit_transform(train.transpose(2, 0, 1).reshape(n_samples * n_points, n_features))
    train_scaled = train_scaled.reshape(n_points, n_samples, n_features).transpose(1, 2, 0)

    n_samples, n_features, n_points = test.shape
    test_scaled = scaler.transform(test.transpose(2, 0, 1).reshape(n_samples * n_points, n_features))
    test_scaled = test_scaled.reshape(n_points, n_samples, n_features).transpose(1, 2, 0)

    if val:
        n_samples, n_features, n_points = test.shape
        val_scaled = scaler.transform(val.transpose(2, 0, 1).reshape(n_samples * n_points, n_features))
        val_scaled = test_scaled.reshape(n_points, n_samples, n_features).transpose(1, 2, 0)

        return train_scaled, test_scaled, val_scaled
    
    return train_scaled, test_scaled, []

def normalize_sample(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

def normalize_data(data):
    
    data_scaled = data.copy()
    n_samples, n_features, n_points = np.array(data).shape
    for i in range(n_samples):
        data_scaled[i] = normalize_sample(data_scaled[i].reshape(n_features*n_points, -1)).reshape(n_features, n_points)

    return data_scaled
    
def featurize_data(data, x, Y: np.ndarray, method = 'logfit_4', aug_factor: int = 1):

    n_samples, n_features, n_points = data.shape
    failed = []
    result_x = []
    result_Y = []

    assert n_samples == len(Y)

    if method.startswith('logfit'):
        num_par = int(method.split('_')[1])
        
        for s in range(n_samples):
            result = np.zeros((n_features, num_par))
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
                print(f'Failed to feturize {x[ind]}, {ind}, {s}')
                failed.append(x[ind])
                continue

        return np.array(result_x), np.array(result_Y), failed
    
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
    
"""def _check_for_irregularity(x, Y, data_save_dict, dataset, features = FEATURES, 
                           scale = False, norm_patient = False, norm_sample = True, method = 'logfit_4',
                           augment_data = False, augment_sample = False, aug_factor: int = 1) -> list:
    
    Scaler = StandardScaler()
    X_data = get_data(data_save_dict, np.tile(x, aug_factor), filter=True, norm_sample=norm_sample, features=features)

    if augment_sample:
        X_data = augmentSample(X_data)

    if augment_data:
        X_data, Y = augmentData(X_data, Y)

    if scale:
        X_data, X_data = scale_data(Scaler, X_data, X_data)

    if norm_patient:
        X_data = normalize_data(X_data)

    _, _, failed = featurize_data(X_data, x.values, Y, method=method)

    failed_set = set(failed)
    print(failed_set)

    x_good = [i for i in x if i not in failed_set]
    dataset_good = dataset[dataset['Patient_id'].isin(x_good)].reset_index(drop=True)

    x = dataset_good['Patient_id']
    Y = dataset_good['D_class']

    return x, Y"""
    
def split_and_process(x, Y, data_save_dict, dataset, features = FEATURES, validation=False,
                      split = 'strat', n_split_val:int = 5, n_split_test: int = 5, 
                      scale_pat = False, scale_batch = False, 
                      norm_patient = False, norm_sample = True, 
                      featurize = True, method = 'logfit_4',
                      augment_data = False, augment_sample = False, aug_factor: int = 1,
                      delimeter:float = 1.0):
    
    assert not (norm_patient and norm_sample)
    assert not (augment_sample and augment_data)
    assert not (not augment_sample and aug_factor != 1), 'With augment_sample False aug_factor must be 1'
    assert delimeter >= 1.0

    Y_train_all, Y_test_all, Y_val_all = [], [], []
    X_train_all, X_test_all, X_val_all = [], [], []
    
    train_val_ids, test_ids, Y_train_val, Y_test, val_ids, Y_val = split_data(x, Y, validation=validation, 
                                                                              split=split, n_split_test=n_split_test, n_split_val=n_split_val)
    
    if validation:
        for i in range(n_split_test*n_split_val):

            Scaler = StandardScaler()

            Y_train_split = np.tile(Y_train_val[i], aug_factor)
            Y_test_split = Y_test[i]
            Y_val_split = Y_val[i]

            X_train_split = get_data(data_save_dict, np.tile(train_val_ids[i], aug_factor), filter=True, norm_sample=norm_sample, features=features, delimeter=1.0)
            X_test_split = get_data(data_save_dict, np.tile(test_ids[i], 1), filter=True, norm_sample=norm_sample, features=features, delimeter=1.0)
            X_val_split = get_data(data_save_dict, np.tile(val_ids[i], 1), filter=True, norm_sample=norm_sample, features=features, delimeter=1.0)
            
            if augment_sample:
                X_train_split = augmentSample(X_train_split)

            if augment_data:
                X_train_split, Y_train_split = augmentData(X_train_split, Y_train_split)

            if scale_batch:
                X_train_split, X_test_split, X_val_split = scale_batch(Scaler, X_train_split, X_test_split, X_val_split)

            if scale_pat:
                X_train_split, X_test_split, X_val_split = scale_patient(X_train_split), scale_patient(X_test_split), scale_patient(X_val_split)

            if norm_patient:
                X_train_split, X_test_split, X_val_split = normalize_data(X_train_split), normalize_data(X_test_split), normalize_data(X_val_split)
        #print(X_train_split.shape, X_test_split.shape)
        
            if featurize:
                X_train_split, Y_train_split, failed1 = featurize_data(X_train_split, np.tile(train_val_ids[i], aug_factor), Y_train_split, method=method)
                X_test_split, Y_test_split, failed2 = featurize_data(X_test_split, np.tile(test_ids[i], 1), Y_test_split, method=method)
                X_val_split, Y_val_split, failed3 = featurize_data(X_val_split, np.tile(val_ids[i], 1), Y_val_split, method=method)

                Y_train_all.append(Y_train_split)
                Y_test_all.append(Y_test_split)
                Y_val_all.append(Y_val_split)
                X_train_all.append(X_train_split)
                X_test_all.append(X_test_split)
                X_val_all.append(X_val_split)
                print(X_train_split.shape, X_test_split.shape, X_val_split.shape, "X")
                print(Y_train_split.shape, Y_test_split.shape, Y_val_split.shape, "Y")
                print(failed1, failed2, failed3)
                #print(*zip(test_ids[i], Y_test_split))
    else:

        train_ids = []
    
        for i in range(n_split_test):

            Scaler = StandardScaler()
            
            train_ids.append(train_val_ids)
            test_ids.append(test_ids)

            Y_train_split = np.tile(Y_train_val[i], aug_factor)
            Y_test_split = Y_test[i]
            np.tile(test_ids[i], aug_factor)
            X_train_split = get_data(data_save_dict, np.tile(train_val_ids[i], aug_factor), filter=True, norm_sample=norm_sample, features=features, delimeter=1.0)
            X_test_split = get_data(data_save_dict, np.tile(test_ids[i], 1), filter=True, norm_sample=norm_sample, features=features, delimeter=1.0)
            
            if augment_sample:
                X_train_split = augmentSample(X_train_split)

            if augment_data:
                X_train_split, Y_train_split = augmentData(X_train_split, Y_train_split)

            if scale_patient:
                X_train_split, X_test_split = scale_patient(X_train_split), scale_patient(X_test_split),

            if scale_batch:
                X_train_split, X_test_split = scale_batch(Scaler, X_train_split, X_test_split)

            if norm_patient:
                X_train_split, X_test_split = normalize_data(X_train_split), normalize_data(X_test_split)
            
            if featurize:
                X_train_split, Y_train_split, failed1 = featurize_data(X_train_split, np.tile(train_val_ids[i], aug_factor), Y_train_split, method=method)
                X_test_split, Y_test_split, failed2 = featurize_data(X_test_split, np.tile(test_ids[i], 1), Y_test_split, method=method)

            Y_train_all.append(Y_train_split)
            Y_test_all.append(Y_test_split)
            X_train_all.append(X_train_split)
            X_test_all.append(X_test_split)
            print(X_train_split.shape, X_test_split.shape, 'X')
            print(Y_train_split.shape, Y_test_split.shape, "Y")
            print(failed1, failed2)
            #print(*zip(test_ids[i], Y_test_split))

    return X_train_all, X_test_all, Y_train_all, Y_test_all, X_val_all, Y_val_all, train_ids, test_ids

def split_data(x, Y, validation = False, split = 'strat',
                      n_split_test: int = 5, n_split_val: int = 5,):
    
    splits = _split_function(x, Y, split=split, n_split=n_split_test,)
    
    train_val_ids, test_ids, Y_train_val, Y_test, val_ids, Y_val = [], [], [], [], [], []
    
    for i in range(n_split_test):
        train_val, test = next(splits)
        print(f'Train_val: {len(train_val)}, test: {len(test)}')

        if validation:
            x_train_val = x[train_val].values
            y_train_val = Y[train_val].values
            splits_val = _split_function(x_train_val, y_train_val, split=split, n_split=n_split_val,)
            for j in range(n_split_val):
                train, val = next(splits_val)
                print(f'Train: {len(train)}, val: {len(val)}')

                train_val_ids.append(x_train_val[train])
                test_ids.append(x[test].values)
                Y_train_val.append(y_train_val[train])
                Y_test.append(Y[test].values)
                val_ids.append(x_train_val[val])
                Y_val.append(y_train_val[val])
                    
        else:
            
            train_val_ids.append(x[train_val].values)
            test_ids.append(x[test].values)
            Y_train_val.append(Y[train_val].values)
            Y_test.append(Y[test].values)

    return train_val_ids, test_ids, Y_train_val, Y_test, val_ids, Y_val

def _split_function(x, Y, split='strat', n_split=5):

    if split == 'strat':
        skf = StratifiedKFold(n_splits=n_split, 
                    #random_state=42, 
                    shuffle=False)
        
    elif split == 'random':
        skf = KFold(n_splits=n_split, 
                    random_state=42, 
                    shuffle=True)

    splits = skf.split(x, Y)

    return splits

### old version
def split_and_process_old(x, Y, data_save_dict, dataset, features = FEATURES,
                      split = 'strat', scale = False, norm_patient = False, norm_sample = True, featurize = True, method = 'logfit_4',
                      augment_data = False, augment_sample = False, aug_factor: int = 1, 
                      n_split: int = 5, train: list = None, test: list = None,
                      delimeter:float = 1.0):
    
    assert not (norm_patient and norm_sample)
    assert not (augment_sample and augment_data)
    assert not (not augment_sample and aug_factor != 1), 'With augment_sample False aug_factor must be 1'

    assert not (n_split == 1 and (train is None or test is None))

    assert delimeter >= 1.0
    
    if n_split > 1:
        if split == 'strat':
            skf = StratifiedKFold(n_splits=n_split, 
                        #random_state=42, 
                        shuffle=False)
            
        elif split == 'random':
            skf = KFold(n_splits=n_split, 
                        random_state=42, 
                        shuffle=True)

        splits = skf.split(x, Y)

    else:
        splits = _oneElemIter(train, test)

    X_train, X_test, Y_train, Y_test = [], [], [], []
    train_ids, test_ids = [], []
    
    for i in range(n_split):
        Scaler = StandardScaler()
        train, test = next(splits)
        print(len(train), len(test))
        
        train_ids.append(x[train].values)
        test_ids.append(x[test].values)

        Y_train_split = np.tile(Y[train].values, aug_factor)
        Y_test_split = Y[test].values
        X_train_split = get_data(data_save_dict, np.tile(x[train], aug_factor), filter=True, norm_sample=norm_sample, features=features, delimeter=1.0)
        X_test_split = get_data(data_save_dict, np.tile(x[test], 1), filter=True, norm_sample=norm_sample, features=features, delimeter=1.0)
        
        if augment_sample:
            X_train_split = augmentSample(X_train_split)

        if augment_data:
            X_train_split, Y_train_split = augmentData(X_train_split, Y_train_split)
        #print(len(train), len(test))
        #print(X_train_split.shape, X_test_split.shape)

        if scale:
            X_train_split, X_test_split = scale_batch(Scaler, X_train_split, X_test_split)

        if norm_patient:
            X_train_split, X_test_split = normalize_data(X_train_split), normalize_data(X_test_split)
        #print(X_train_split.shape, X_test_split.shape)
        
        if featurize:
            X_train_split, Y_train_split, failed1 = featurize_data(X_train_split, np.tile(x[train], aug_factor), Y_train_split, method=method)
            X_test_split, Y_test_split, failed2 = featurize_data(X_test_split, np.tile(x[test], 1), Y_test_split, method=method)
        #print(X_train_split.shape, X_test_split.shape)

        Y_train.append(Y_train_split)
        Y_test.append(Y_test_split)
        X_train.append(X_train_split)
        X_test.append(X_test_split)
        print(X_train_split.shape, X_test_split.shape, 'X')
        print(Y_train_split.shape, Y_test_split.shape, "Y")
        print(failed1, failed2)
        print(*zip(x[test], Y_test_split))

    return X_train, X_test, Y_train, Y_test, train_ids, test_ids

def _oneElemIter(a, b):
        yield(a, b)