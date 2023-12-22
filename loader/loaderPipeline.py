import os.path

from loader.hisarLoader import *
from loader.RML2018a import *
from utils.commonUtils import *
from rich.progress import track
from typing import *
from utils.commonUtils import onehot_encoder


def hisar_loader_pipeline(length: int):
    console = Console()
    console.print("data loader process starts")

    if not os.path.exists(HISAR_PROCESSED):
        # 如果不存在，则创建文件夹
        os.makedirs(HISAR_PROCESSED)
        os.makedirs(HISAR_PROCESSED_TRAIN)
        os.makedirs(HISAR_PROCESSED_TEST)
    else:
        console.print("the processed dir already exists", style="bold red")
        return
    console.print("processed dir create successfully", style="bold green")
    hisar_data = HisarLoader(path_conf=HISAR)
    data_train = hisar_data.get_data_train(length)
    label_train = hisar_data.get_label_train(length)
    snr_train = hisar_data.get_snr_train(length)

    filter_snr(console, HISAR_TRAIN_SNR_INFO, data_train, label_train, snr_train, HISAR_PROCESSED_TRAIN,
               HISAR_PROCESSED_TRAIN_FILE_NAME,
               2, 25)

    del data_train
    del label_train
    del snr_train
    data_test = hisar_data.get_data_test(length)
    label_test = hisar_data.get_label_test(length)
    snr_test = hisar_data.get_snr_test(length)
    filter_snr(console, HISAR_TEST_SNR_INFO, data_test, label_test, snr_test, HISAR_PROCESSED_TEST, "data{}.hdf5",
               2, 25)

    console.print("congrats! your data is ready", style="bold green")




def RML_2018a_processor():
    console = Console()
    retrieve = data_retrieve(RML_2018A_FILTERED, "hdf5")
    filtered_label_path = os.path.join(RML_2018A_FILTERED, "2018filteredClass.json")

    with open(RML_2018A_LABEL_META) as f:
        origin_label = json.load(f)
    print(origin_label)
    with open(filtered_label_path) as f:
        filtered_label = json.load(f)
    print(filtered_label)
    filtered_label_encoder = onehot_encoder(filtered_label)
    filtered_data_set = []
    filtered_label_set = []
    filtered_snr_set = []
    for index, data in enumerate(retrieve):
        length = data['label'].shape[0]
        for i in track(range(length)):
            current_label = onehot2label(data['label'][i], origin_label)
            if current_label in filtered_label:
                onehot_label = filtered_label_encoder[current_label]
                filtered_data_set.append(data['data'][i])
                filtered_label_set.append(onehot_label)
                filtered_snr_set.append(data["snr"][i])
        console.print("label is filtered successfully")
        with h5py.File(RML_2018A_PROCESSED_FILE_LABEL_FILTERED.format(index), 'w') as f:
            f.create_dataset('data', data=filtered_data_set)
            f.create_dataset('label', data=filtered_label_set)
            f.create_dataset('snr', data=filtered_snr_set)
        console.print("new data is stored successfully")


def HISAR_processed():
    retrieve = data_retrieve(HISAR_PROCESSED_TEST, "hdf5", index_col=0)
    candidate_label_path = os.path.join(BASE_CONF, "2019filteredClass.json")
    with open(candidate_label_path, 'r') as f:
        filtered_label = json.load(f)
    filtered_label: dict[int, str] = {int(k): v for k, v in filtered_label.items()}
    filtered_label_keys = list(filtered_label.keys())
    onehot_label = onehot_encoder(filtered_label_keys)
    for index, data in enumerate(retrieve):
        indices = np.isin(data['label'][:], filtered_label_keys)
        indices = indices.reshape(-1)
        filtered_labels = data['label'][indices]
        filtered_data = data['data'][indices]
        filtered_snr = data['snr'][indices]
        if not os.path.exists(HISAR_FILTERED_LABEL):
            os.makedirs(HISAR_FILTERED_LABEL)
        print("create dir successfully")
        with h5py.File(HISAR_FILTERED_LABEL_TEST_FILE.format(index), 'w') as f:
            f.create_dataset("data", data=filtered_data)
            f.create_dataset("label", data=filtered_labels)
            f.create_dataset('snr', data=filtered_snr)
    print("filtered label successfully")


def HISAR_onehot_converter():
    file_path = os.path.join(BASE_PATH, "processedLabel")
    retrieve = data_retrieve(file_path, "hdf5")
    candidate_label_path = os.path.join(BASE_CONF, "2019filteredClass.json")
    with open(candidate_label_path, 'r') as f:
        filtered_label = json.load(f)
    filtered_label: dict[int, str] = {int(k): v for k, v in filtered_label.items()}
    filtered_label_keys = list(filtered_label.keys())
    onehot_label = onehot_encoder(filtered_label_keys)
    print(onehot_label)
    for ind, data in enumerate(retrieve):
        print(data.keys())
        onehot_label_array = []
        for label in data['label']:
            onehot_label_array.append(onehot_label[label[0]])
        np.array(onehot_label_array)
        with h5py.File(HISAR_FILTERED_ONEEHOT_TRAIN_FILE.format(ind), 'w') as f:
            f.create_dataset("data", data=data['data'])
            f.create_dataset("label", data=np.array(onehot_label_array))
            f.create_dataset("snr", data=data['snr'])




