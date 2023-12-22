import json
import time

import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os.path
from rich import print
from conf import *
from utils.commonUtils import onehot2label
from multiprocessing import Pool, Value, Lock
from rich.progress import Progress

task_num = 0
task_value = Value('i', 0)
lock = Lock()


def process_data(*args):
    iq_data,label_pack = args
    i,label = label_pack
    with lock:
        task_value.value += 1
    # 使用面向对象的方法创建图形和轴
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(iq_data[:, 0], iq_data[:, 1], 'o')
    ax.grid(False)
    ax.axis('off')

    # 保存到内存中的文件对象
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', dpi=300)
    plt.close(fig)  # 关闭图形以节省资源

    # 使用内存中的数据创建PIL图像
    im = Image.open(buf)
    im_resized = im.resize((224, 224), Image.LANCZOS)

    # 保存调整大小后的图像
    label_dir = os.path.join(RML_RGB_IMG, "{}".format(label))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    img_file_path = os.path.join(label_dir, "{}.jpg".format(i))
    im_resized.save(img_file_path)
    # print("image {} complete".format(i))



def iq2RGB(process_num=20):
    if not os.path.exists(RML_RGB_IMG):
        os.makedirs(RML_RGB_IMG)
        print("folder creates successfully")

    filtered_label_path = RML_2018A_PROCESSED_FILE_LABEL_FILTERED_COMPLETE
    h5_data = h5py.File(filtered_label_path, 'r')

    with open(RML_2018_SPECIFIC_LABEL, 'r') as f:
        label = json.load(f)

    # convert the onehot label and save the data id and label string in data_label_array
    # which is useful to place them in the right dir
    if not os.path.exists(RML_LABEL_CACHE):
        data_label_array = []
        for id, onehot_label in enumerate(h5_data['label']):
            data_label_array.append((id, onehot2label(onehot_label, label)))
        np.save(RML_LABEL_CACHE, np.array(data_label_array))
    label_array = np.load(RML_LABEL_CACHE)
    global task_num
    task_num = len(label_array)

    print("the image record task starts")
    packed_data = zip(h5_data['data'][:], label_array)
    with Pool(process_num) as p:
        p.starmap(process_data, packed_data)
        with Progress() as progress:
            task = progress.add_task("[green]Processing...", total=task_num)
            while not progress.finished:
                with lock:
                    progress.update(task, advance=task_value)
                    time.sleep(1)






