import os.path

LOGO = """ 

 _______                 _            _       
(_______)               | |          | |      
 _____   ____  ___ _   _| |      ____| | _    
|  ___) / _  |/___) | | | |     / _  | || \\   
| |____( ( | |___ | |_| | |____( ( | | |_) )  
|_______)_||_(___/ \\__  |_______)_||_|____/   
                  (____/                      

"""
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_CONF = os.path.join(BASE_PATH, "conf")

HISAR_BASE = os.path.join(BASE_PATH, "dataset", "2019", "HisarMod2019.1")
HISAR_TRAIN = os.path.join(HISAR_BASE, "Train")
HISAR_TEST = os.path.join(HISAR_BASE, "Test")

HISAR = {"train_data_path": os.path.join(HISAR_TRAIN, "train_data.csv"),
         "train_label_path": os.path.join(HISAR_TRAIN, "train_labels.csv"),
         "train_snr_path": os.path.join(HISAR_TRAIN, "train_snr.csv"),
         "test_data_path": os.path.join(HISAR_TEST, "test_data.csv"),
         "test_label_path": os.path.join(HISAR_TEST, "test_labels.csv"),
         "test_snr_path": os.path.join(HISAR_TEST, "test_snr.csv"),
         }


HISAR_PROCESSED = os.path.join(BASE_PATH, "HISARProcessed")
HISAR_PROCESSED_TRAIN = os.path.join(HISAR_PROCESSED, "train")
HISAR_PROCESSED_TRAIN_FILE_NAME = "data{}.hdf5"
HISAR_PROCESSED_TEST = os.path.join(HISAR_PROCESSED, "test")

HISAR_FILTERED_LABEL = os.path.join(BASE_PATH, "ProcessedLabel")
HISAR_FILTERED_LABEL_TRAIN_FILE = os.path.join(HISAR_FILTERED_LABEL,
                                               "hisar_filtered_train_label{}.hdf5")
HISAR_FILTERED_LABEL_TEST_FILE = os.path.join(HISAR_FILTERED_LABEL, "hisar_filtered_test_label{}.hdf5")

HISAR_FILTERED_ONEEHOT_TRAIN_FILE =os.path.join(HISAR_FILTERED_LABEL,"hisar_filtered_train_onehot{}.hdf5")
HISAR_FILTERED_ONEEHOT_TEST_FILE = os.path.join(HISAR_FILTERED_LABEL,"hisar_filtered_test_onehot{}.hdf5")

HISAR_TRAIN_SNR_INFO = "working on train set:filtering range of signal-to-noise"
HISAR_TEST_SNR_INFO = "working on test set:filtering range of signal-to-noise"

RML_2018A = os.path.join(BASE_PATH, "dataset", "2018")
RML_2018A_DATA = os.path.join(RML_2018A, "GOLD_XYZ_OSC.0001_1024.hdf5")
RML_2018A_LABEL_META = os.path.join(RML_2018A, "classes-fixed.json")

# filter range signal-to-noise
RML_2018A_FILTERED = os.path.join(BASE_PATH, "RML2018Processed")
RML_2018A_PROCESSED_FILE = os.path.join(RML_2018A_FILTERED, "RML2018a_{}.hdf5")

# filter the label
RML_2018_SPECIFIC_LABEL = os.path.join(BASE_PATH, "conf", "2018filteredClass.json")
RML_2018A_PROCESSED_FILE_LABEL_FILTERED = os.path.join(RML_2018A_FILTERED, "RML2018a_label_filtered{}.hdf5")
RML_TASK_MESSAGE = "working on process dataset"
RML_RGB_IMG = os.path.join(BASE_PATH, "RML2018Img")
RML_LABEL_CACHE = os.path.join(RML_2018A_FILTERED, "label_cache.npy")

# test conf
RML_2018A_PROCESSED_FILE_LABEL_FILTERED_COMPLETE = os.path.join(RML_2018A_FILTERED, "RML2018a_label_filtered0.hdf5")
