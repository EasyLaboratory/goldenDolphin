import json
import os.path

import numpy as np
import h5py
from rich.console import Console
from conf import RML_2018A_DATA
from conf import RML_2018A_LABEL_META
from conf import RML_2018A_FILTERED
from conf import RML_2018A_PROCESSED_FILE


class RML2018a:
    def __init__(self, console: Console = None, data_path=RML_2018A_DATA, label_path=RML_2018A_LABEL_META,
                 batch_size=100000):
        self.console = console
        self.f: h5py.File = h5py.File(data_path, 'r')
        self.label = RML2018a.get_label(label_path)
        self.batch_size = batch_size

    def describe(self):
        print("root contains groups: ", self.f.keys())
        print("data shape: ", self.f['X'].shape, "dtype: ", self.f['X'].dtype)
        print("-----------------------------------------------------------------------")
        print("label shape: ", self.f['Y'].shape, "dtype: ", self.f['Y'].dtype)
        print("-----------------------------------------------------------------------")
        print("snr shape: ", self.f['Z'].shape, "dtype: ", self.f['Z'].dtype)
        print("-----------------------------------------------------------------------")

    def filter_snr(self, low: float, high: float):
        totalSize = self.f['X'].shape[0]
        counter = 0
        with self.console.status("[bold green]working on the filtering range of signal-to-noise") as status:
            for i in range(0, totalSize, self.batch_size):
                condition = (low <= self.f['Z'][i:self.batch_size]) & (self.f['Z'][i:self.batch_size] <= high)
                indices = np.where(condition)
                data_feature = self.f['X'][i:self.batch_size]
                label = self.f['Y'][i:self.batch_size]
                snr = self.f['Z'][i:self.batch_size]
                filtered_data = data_feature[indices[0]]
                filtered_label = label[indices[0]]
                filtered_snr = snr[indices[0]]
                if not os.path.exists(RML_2018A_FILTERED):
                    os.makedirs(RML_2018A_FILTERED)
                    self.console.print("processed data create successfully", style="bold green")

                with h5py.File(RML_2018A_PROCESSED_FILE.format(i), 'w') as f:
                    f.create_dataset('data', data=filtered_data)
                    f.create_dataset('label', data=filtered_label)
                    f.create_dataset('snr', data=filtered_snr)
                    self.console.print("filtered data is recorded to", RML_2018A_PROCESSED_FILE, style="bold green")
                status.console.print("task {} complete".format(counter))
                counter += 1

    def filter_modulation_type(self):
        print(self.f.keys())
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def __getitem__(self, item):
        return (self.f['X'][item], self.f['Y'][item],
                self.f['Z'][item])

    def close(self):
        self.f.close()

    @staticmethod
    def get_label(label_path=RML_2018A_LABEL_META):
        with open(label_path, 'r') as f:
            return json.load(f)

