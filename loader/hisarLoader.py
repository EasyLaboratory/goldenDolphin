import h5py
import numpy as np
from loader.commonImport import *
from rich.console import Console
import re


class HisarLoader:
    def __init__(self, path_conf: dict[str, str] = HISAR):
        self._data_train_path = path_conf["train_data_path"]
        self._label_train_path = path_conf["train_label_path"]
        self._snr_train_path = path_conf["train_snr_path"]
        self._data_test_path = path_conf["test_data_path"]
        self._label_test_path = path_conf["test_label_path"]
        self._snr_test_path = path_conf["test_snr_path"]
        pass

    def get_data_train(self, chunk_size=1000):
        return pd.read_csv(self._data_train_path, header=None, chunksize=chunk_size)

    def get_label_train(self, chunk_size=1000):
        data_types = {0: np.int8}
        return pd.read_csv(self._label_train_path, header=None, chunksize=chunk_size, dtype=data_types, names=["label"])

    def get_snr_train(self, chunk_size=1000):
        data_types = {0: np.int8}
        return pd.read_csv(self._snr_train_path, header=None, chunksize=chunk_size, dtype=data_types, names=["snr"])

    def get_data_test(self, chunk_size=1000):
        return pd.read_csv(self._data_test_path, header=None, chunksize=chunk_size)

    def get_label_test(self, chunk_size=1000):
        return pd.read_csv(self._label_test_path, header=None, chunksize=chunk_size, names=["label"])

    def get_snr_test(self, chunk_size=1000):
        return pd.read_csv(self._snr_test_path, header=None, chunksize=chunk_size, names=["snr"])

    def get_data_train_nrow(self, row_size=500):
        return pd.read_csv(self._data_train_path, header=None, nrows=row_size)

    def get_label_train_nrow(self, row_size=500):
        return pd.read_csv(self._label_train_path, header=None, nrows=row_size)

    def get_snr_train_nrow(self, row_size=500):
        return pd.read_csv(self._snr_test_path, header=None, nrows=row_size)


def file_path_generator(console: Console, base_path: str, file_name: str, file_id: int):
    if re.search(r'{ *}', file_name) is None:
        console.print("the file name should include {}", style="bold red")
        raise Exception
    return os.path.join(base_path, file_name.format(file_id))


def filter_snr(console: Console, status_info: str, data, label, snr, base_path: str,
               data_file_name: str,
               lower: float,
               upper: float):
    step = 0

    with console.status("[bold green]" + status_info) as status:
        for snr_point, label_point, data_point in zip(snr, label, data):
            # Filter out data within a certain signal-to-noise ratio range
            condition = (snr_point["snr"] <= upper) & (snr_point["snr"] >= lower)
            filtered_snr: pd.DataFrame = snr_point[condition]

            if filtered_snr.empty:
                continue
            filtered_label: pd.DataFrame = label_point[condition]
            filtered_data: pd.DataFrame = data_point[condition]

            # the data format in the dataframe should a+bi to keep the data
            # processing consistence convert 0 to 0+0i.
            filtered_data:pd.DataFrame = filtered_data.replace("0","0+0i")

            # split (1,1024)data to (2,1024),the iq_vector shape
            iq_vector: np.ndarray = regex_search(filtered_data)

            # store the data in hdf5 file
            data_integrated_path = file_path_generator(console, base_path, data_file_name, step)
            with h5py.File(data_integrated_path, 'w') as f:
                numpy_label = filtered_label.to_numpy()
                numpy_snr = filtered_snr.to_numpy()
                f.create_dataset("data", data=iq_vector)
                f.create_dataset("label", data=numpy_label)
                f.create_dataset("snr", data=numpy_snr)

            status.console.log("data batch{} complete".format(step))
            step += 1


def regex_search(df: pd.DataFrame, pattern=r"([+-]?\d+\.?\d*(?:e[+-]?\d+)?)") -> np.array:
    iq_vector = []
    for index, row in df.iterrows():
        try:
            res: pd.DataFrame = row.str.extractall(pattern)
        except ValueError:
            pd.DataFrame({"origin":row}).to_csv("cache")
            print("没找到匹配")
        else:
            vector_0 = res.xs(0, level=1).values.astype(float).flatten()
            vector_1 = res.xs(1, level=1).values.astype(float).flatten()
            if vector_0.shape != vector_1.shape:
                pd.DataFrame({"origin": row}).to_csv("cache.csv")
                pd.DataFrame({"vec_0":vector_0}).to_csv("vec0.csv")
                pd.DataFrame({"vec_1": vector_1}).to_csv("vec1.csv")
                raise Exception("维度不一致")
            combined_vec = np.vstack((vector_0, vector_1)).T
            iq_vector.append(combined_vec)
    return np.array(iq_vector)
