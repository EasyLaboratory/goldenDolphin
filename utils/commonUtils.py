import os
import h5py
import numpy as np
import pandas as pd
from typing import *


def data_retrieve(data_dir: str, file_type: str, **kwargs):
    files = os.listdir(data_dir)
    files = [file for file in files if file.endswith(file_type)]
    print(files)
    sorted_files = sorted(files)
    if file_type == "hdf5":
        for file in sorted_files:
            file_path = os.path.join(data_dir, file)
            with h5py.File(file_path, 'r') as f:
                yield f
    elif file_type == "csv":
        for file in sorted_files:
            file_path = os.path.join(data_dir, file)
            yield pd.read_csv(file_path, index_col=kwargs["index_col"])


def onehot2label(onehot_label: np.ndarray, label: list[str]):
    index = np.where(onehot_label == 1)[0][0]
    return label[index]


def onehot_encoder(label: list[Union[str, int]]) -> dict:
    label2onehot = {}
    onehot_label_length = len(label)
    for ind, label_i in enumerate(label):
        onehot_label = np.zeros(onehot_label_length)
        onehot_label[ind] = 1
        label2onehot[label_i] = onehot_label
    return label2onehot


def onehot2label(onehot_label: np.ndarray, label: List[str]):
    index = np.where(onehot_label == 1)[0][0]
    return label[index]
