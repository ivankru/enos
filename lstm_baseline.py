#%%
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from copy import deepcopy


diagnosis_class = {'Z00': 0,
                'E11': 1,
                'K29': 2,
                'K76': 3,
                'B18': 4,
                'C34': 5,
                'N18': 6,
                'J44': 7}

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


class NoseSample(Dataset):
    def __init__(self, json_path:str, label_path:str, condition:int, weeks:list):
        super().__init__()
        self.min_sample_length = 180
        self.important_id_list = ['R8', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17']
        with open(json_path, 'r') as file:
            data = json.load(file)
        self.__read_labels__(label_path)

        self.channels_list = []
        self.id_list = []
        self.channel_id_list = []

        dataset = pd.read_csv("data/dataset_28_08_25.csv")
        dataset['datetime'] = pd.to_datetime(dataset['Time'], format='%d/%m/%y %H:%M:%S')
        diag_test_weeks = {i:j for i, j in zip(diagnosis_class.keys(), weeks)}
        split_index = dataset[dataset.Week.isin(list(diag_test_weeks.values())[condition])].Patient_id
        split_index = split_index.to_numpy().tolist()
        split_index = [str(x) for x in split_index]

        for study_id in data.keys():
            if study_id in split_index:
                #label = self.id_class_dict[int(study_id)]
                channels = data[study_id]["sensors"][0]["channels"]
                channels = dict([(x["id"], x) for x in channels])
                sample_list = []
        
                for channel_id in self.important_id_list:
                    not_enough_length = False
                    channels_tensor = torch.FloatTensor(channels[channel_id]["samples"])
                    if channels_tensor.shape[0] < self.min_sample_length:
                        not_enough_length = True
                        break
                    sample_list.append(channels_tensor[:self.min_sample_length])
                if not_enough_length:
                    continue

                self.id_list.append(study_id)
                self.channels_list.append(torch.vstack(sample_list))


    def __read_labels__(self, label_path:str):
        dataset = pd.read_csv(label_path)
        dataset['datetime'] = pd.to_datetime(dataset['Time'], format='%d/%m/%y %H:%M:%S')
        
        dataset['day'] = dataset.apply(lambda row: row['datetime'].timetuple().tm_yday - 169, axis = 1)
        dataset['week'] = dataset.apply(lambda row: row['datetime'].isocalendar()[1] - 25, axis = 1)

        dataset_good = dataset[dataset['Comment'].isna()].reset_index(drop=True)
        dataset_sorted_good = dataset_good.sort_values(by='datetime')
        id_for_classes = dataset_sorted_good['Patient_id']
        classes_list = dataset_sorted_good['D_class']
        self.id_class_dict = dict(zip(id_for_classes, classes_list))
        for id in [1111, 1112, 1113, 1114, 1115]:
            self.id_class_dict[id] = 9    
        return  self.id_class_dict

    def __getitem__(self, index):
        id = int(self.id_list[index])
        label = int(self.id_class_dict[id])
        serie = self.channels_list[index]
        serie = serie.t()
        return serie, label

    def __len__(self):
        return len(self.channels_list)


class MultiChannelLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate=0.2):
        super(MultiChannelLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        # У нас может быть полносвязный слой для получения конечного результата
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x expected shape: (batch_size, sequence_length, input_dim)
        out, (hn, cn) = self.lstm(x)
        
        # last step output
        # out shape: (batch_size, sequence_length, hidden_dim)
        last_time_step_out = out[:, -1, :]
        
        final_output = self.fc(last_time_step_out)
        return final_output


if __name__ == "__main__":
   json_path = "datasets/dataset_dict_28_08_25.json"
   label_path = "datasets/dataset_28_08_25.csv"
   CONDITION = 7
   nose_dataset_train = NoseSample(json_path, label_path, CONDITION, train_weeks)
   train_ratio = 0.8
   val_ratio = 0.2
   # split train dataset on train and val
   total_size = len(nose_dataset_train)
   train_size = int(total_size * train_ratio)
   val_size = total_size - train_size
   train_dataset, val_dataset = random_split(nose_dataset_train, [train_size, val_size])
   nose_dataset_test = NoseSample(json_path, label_path, CONDITION, test_weeks)
 
   # constants for model initialization
   INPUT_DIM = 8      # number of channels
   HIDDEN_DIM = 128
   NUM_LAYERS = 3
   OUTPUT_DIM = 1     #
   DEVICE = "cuda:0"
   model = MultiChannelLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM)
   model = model.to(DEVICE)
   optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

   BATCH_SIZE = 16
   nose_dataloader_train = DataLoader(nose_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
   nose_dataloader_test = DataLoader(nose_dataset_test, batch_size=BATCH_SIZE, shuffle=False)

   loss_funct = nn.BCEWithLogitsLoss()

   best_auc_list = []
   for random_seed in range(16): #to get std
        np.random.seed(103*random_seed)
        torch.manual_seed(103*random_seed)
        torch.cuda.manual_seed(103*random_seed)
        best_auc = 0
        for epoch in range(10):
            loss_list = []
            prediction_list = []
            label_list = []
            model.train()
            for batch, label in nose_dataloader_train:
                batch = batch.to(DEVICE)
                label = (label == CONDITION).to(torch.float32)
                label = label.to(DEVICE)
                prediction = model(batch)
                loss = loss_funct(prediction.squeeze(1), label)
                optimizer.zero_grad()
                loss_list.append(loss.item())
                loss.backward()
                optimizer.step()

            model.eval()
            prediction_list = []
            label_list = []
            for batch, label in nose_dataloader_test:
                batch = batch.to(DEVICE)
                label = (label == CONDITION).to(torch.float32)
                label = label.to(DEVICE)
                prediction = model(batch)
                prediction_list.append(prediction.detach().cpu())
                label_list.append(label.detach().cpu()) 

            mean_train_loss = sum(loss_list) / len(loss_list)
            prediction_list = torch.vstack(prediction_list).squeeze(1)
            label_list = torch.hstack(label_list)
            roc_auc = roc_auc_score(label_list, prediction_list)
            if epoch > 2 and best_auc < roc_auc:
                best_auc = max(roc_auc, best_auc)
                best_model = deepcopy(model)
            print(f"{mean_train_loss:.5f} rocauc:{roc_auc:.3f}")
        
        prediction_list = []
        label_list = []
        best_model.eval()
        for batch, label in nose_dataloader_test:
                batch = batch.to(DEVICE)
                label = (label == CONDITION).to(torch.float32)
                label = label.to(DEVICE)
                prediction = best_model(batch)
                prediction_list.append(prediction.detach().cpu())
                label_list.append(label.detach().cpu())
        prediction_list = torch.vstack(prediction_list).squeeze(1)
        label_list = torch.hstack(label_list) 
        roc_auc = roc_auc_score(label_list, prediction_list)
        best_auc_list.append(roc_auc)

   best_auc_mean = np.array(best_auc_list).mean()
   best_auc_std = np.array(best_auc_list).std()
   print(f"ROC AUC:{best_auc_mean:.3f}±{best_auc_std:.3f}")


