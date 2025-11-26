import pandas as pd
import numpy as np
import json


def __read_labels__(label_path:str):
    dataset = pd.read_csv(label_path)
    # Например датасет для здоровых и больных диабетом - берем данные с меткой 0 и 1
    #dataset['datetime'] = pd.to_datetime(dataset['Time'], format='%m/%d/%Y %H:%M:%S')
    dataset['datetime'] = pd.to_datetime(dataset['Time'], format='%d/%m/%y %H:%M:%S')
    
    dataset['day'] = dataset.apply(lambda row: row['datetime'].timetuple().tm_yday - 169, axis = 1)
    dataset['week'] = dataset.apply(lambda row: row['datetime'].isocalendar()[1] - 25, axis = 1)

    # for week, df  in dataset.groupby('week'):
    #     print(week)
    #     print(df.index)

    dataset_good = dataset[dataset['Comment'].isna()].reset_index(drop=True)
    dataset_sorted_good = dataset_good.sort_values(by='datetime')
    id_for_classes = dataset_sorted_good['Patient_id']
    classes_list = dataset_sorted_good['D_class']
    id_class_dict = dict(zip(id_for_classes, classes_list))
    for id in [1111, 1112, 1113, 1114, 1115]:
        id_class_dict[id] = 9    
    return  dataset_sorted_good, id_class_dict


def train_test_split(json_path, csv_path):
    dataset, id_class_dict = __read_labels__(csv_path)
    
    inverted_diagnosis_class = {0: 'Здоров; Z00',
        1: 'Сахарный диабетом II 2 типа; E11',
        2: 'Гастрит и дуоденит; K29',
        3: 'Неалкогольная жировая болезнь печени; К76',
        4: 'Гепатит В и/или С; B18',
        5: 'ЗНО лёгких; C34',
        6: 'Хроническая почечная недостаточность; N18',
        7: 'Хроническая обструктивная болезнью лёгких; J44'}

    diagnosis_class = {value: key for key, value in inverted_diagnosis_class.items()}

    #'Здоров; Z00'
    test_weeks0 = [{"week":10, "number_of_positive":34}]
    #'Сахарный диабетом II 2 типа; E11'
    test_weeks1 = [{"week":7, "number_of_positive":21}, {"week":8, "number_of_positive":9}]
    #'Гастрит и дуоденит; K29'
    test_weeks2 = [ {"week":6, "number_of_positive":20}, {"week":10, "number_of_positive":10}]
    #'Неалкогольная жировая болезнь печени; К76'
    test_weeks3 = [{"week":9, "number_of_positive":37}]
    #'Гепатит В и/или С; B18'
    test_weeks4 = [{"week":6, "number_of_positive":25}, {"week":10, "number_of_positive":10}]
    #'ЗНО лёгких; C34'
    test_weeks5 = [{"week":10, "number_of_positive":30}]
    #'Хроническая почечная недостаточность; N18'
    test_weeks6 = [{"week":8, "number_of_positive":8}, {"week":9, "number_of_positive":22}]
    #'Хроническая обструктивная болезнью лёгких; J44'
    test_weeks7 = [{"week":9, "number_of_positive":25}, {"week":10, "number_of_positive":5}]

    test_weeks_dict = {0:test_weeks0, 1:test_weeks1, 2:test_weeks2,
                       3:test_weeks3, 4:test_weeks4, 5:test_weeks5,
                       6:test_weeks6, 7:test_weeks7
                      }
    
    train_test_split_dict = {}
    for class_index in range(8):
        test_weeks = test_weeks_dict[class_index]
        diag_test_weeks = {i:j for i, j in zip(diagnosis_class.keys(), test_weeks)}

        test_ind_list = []
        #select random "number_of_positive" element from the data within a "week" for the test
        #all negatives used in test
        for item in test_weeks:
            week = item["week"] - 1
            number_of_positive = item["number_of_positive"]
            patient_ind = dataset[dataset.week.isin([week])].Patient_id
            test_Y = dataset[dataset.week.isin([week])].D_class
            positive_ind = patient_ind[test_Y.isin([class_index])] 
            negative_ind = patient_ind[~test_Y.isin([class_index])] 
            positive_ind = positive_ind.sample(number_of_positive)
            test_ind_list.append(pd.concat([positive_ind, negative_ind]))
        test_ind = pd.concat(test_ind_list).tolist()
        #the train index are the other patients not included in the test
        train_ind = dataset[~dataset.Patient_id.isin(test_ind)].Patient_id
        train_ind = train_ind.tolist()
        diagnosis = inverted_diagnosis_class[class_index]
        diagnosis_code = diagnosis[-3:]
        train_test_split_for_diag = {"diagnosis":diagnosis, "diagnosis_code":diagnosis_code, "train_index":train_ind, "test_index":test_ind}
        train_test_split_dict.update({class_index:train_test_split_for_diag})

    return train_test_split_dict


if __name__ == "__main__":
    json_path = "data/dataset_dict_28_08_25.json"
    csv_path = "data/dataset_28_08_25.csv"
    np.random.seed(42)

    train_test_split_dict = train_test_split(json_path, csv_path)
    file_save_path = "data/train_test_split.json"
    with open(file_save_path, "w") as file:
        json.dump(train_test_split_dict, file, indent=4) 