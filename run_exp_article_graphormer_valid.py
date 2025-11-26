from transformers.models.graphormer.collating_graphormer import GraphormerDataCollator, preprocess_item
from transformers import GraphormerForGraphClassification, set_seed

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch


from datasets import load_dataset, DatasetDict
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
from transformers import GraphormerForGraphClassification, TrainingArguments, Trainer, TrainerCallback
from sklearn.metrics import roc_auc_score
import numpy as np

import numpy as np
import pandas as pd
import math

import warnings
warnings.filterwarnings('ignore')

from utils import *
from graph_utils import *

norm_sample = False
scale_pat = False

classes = [0, 1, 2, 3, 4, 5, 6, 7]

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

dataset = pd.read_csv("../datasets/dataset_28_08_25.csv")
dataset['datetime'] = pd.to_datetime(dataset['Time'], format='%d/%m/%y %H:%M:%S')

n_samples = dataset.Patient_id.shape[0]
print(f'Dataset has {n_samples} patients')

with open('../datasets/dataset_dict_28_08_25.json') as infile:
    data_save_dict = json.load(infile)

n_splits = 5

rocs_df = pd.DataFrame(columns = ['Diag'] + [str(i) for i in range(1, n_splits + 1)] + ['mean', 'std'])

for i, cl in enumerate(classes):

    rocs_class = []
    class_name = list(diagnosis_class.keys())[i]
    print(class_name)

    train_val_ind = np.array(dataset[dataset.Week.isin(list(diag_train_weeks.values())[i])].Patient_id)

    test_ind = np.array(dataset[dataset.Week.isin(list(diag_test_weeks.values())[i])].Patient_id)
    Y_test = dataset[dataset.Week.isin(list(diag_test_weeks.values())[i])].D_class
    targets_test = [0 if t == cl else 1 for t in Y_test]
    X_test = get_data(data_save_dict, np.tile(test_ind, 1), filter=False, norm_sample=norm_sample, features=FEATURES, delimeter=1.0)

    for random_seed in range(n_splits):
        np.random.seed(103*random_seed)

        #randomly select 15% of train subset for validation
        
        n_val = int(train_val_ind.size * 0.15)
        indices = np.random.choice(train_val_ind.size, size=n_val, replace=False)
        val_label = train_val_ind[indices]
        train_label = np.delete(train_val_ind, indices, axis=0)

        Y_train = np.array([0 if i == cl else 1 for i in dataset[dataset.Patient_id.isin(train_label)].D_class])
        Y_val = np.array([0 if i == cl else 1 for i in dataset[dataset.Patient_id.isin(val_label)].D_class]) 

        X_train = get_data(data_save_dict, np.tile(train_label, 1), filter=False, norm_sample=False, features=FEATURES, delimeter=1.0)
        X_val = get_data(data_save_dict, np.tile(val_label, 1), filter=False, norm_sample=False, features=FEATURES, delimeter=1.0)

        ds_dict = make_graphormer_datasetdict((X_train, X_val, X_test), (Y_train, Y_val, Y_test), make_bins=True)

        ds = ds_dict

        print({k: len(v) for k,v in ds.items()})

        def remat(d): return d.select(range(len(d)))
        ds = DatasetDict({k: remat(v) for k, v in ds.items()})

        set_seed(42)

        model = GraphormerForGraphClassification.from_pretrained(
        "clefourrier/pcqm4mv2_graphormer_base",
        num_classes=2,
        ignore_mismatched_sizes=True,
        )
        
        collator = GraphormerDataCollator(on_the_fly_processing=True)  # since we precomputed features

        args = TrainingArguments(
            output_dir="out",
            learning_rate=8e-5,
            lr_scheduler_type="cosine",
            num_train_epochs=30,
            #num_train_epochs=10,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            auto_find_batch_size=True,
            warmup_steps=300,

            fp16=True, 

            remove_unused_columns=False,
            dataloader_num_workers=12,

            logging_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy="epoch",

            load_best_model_at_end=True,
            metric_for_best_model="roc_auc",
            greater_is_better=True,
            report_to=[],
            seed=42
        )

        trainer = MyTrainer(
            model=model,
            args=args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[EpochPrinter()], 
        )

        print(len(ds["train"]), len(ds["validation"])) 
        train_results = trainer.train()
        val_metrics = trainer.evaluate(eval_dataset=ds['test'])                     
        print(f"Final (val) for {class_name}:", val_metrics)  
        val_roc = val_metrics['eval_roc_auc']
        rocs_class.append(val_roc)        


    mean = np.mean(rocs_class)
    std = np.std(rocs_class, ddof = 1)
    rocs_df.loc[len(rocs_df.index)] = [class_name] + [*rocs_class] + [mean, std]
    print(f'{mean} +- {std} for {class_name}')

rocs_df.to_csv('exp_binary_article_graphormer_valid_split_weeks.csv', index=False)
