# Data Loader
Data loader is a python module for data preprocessing. In AI model training process,the dataset 
needs to be preprocessed with some purposes such as filtering data, change the datatype and shape
to fit the AI model.  

A good beginning is half the battle. But the data preprocessing is always a tricky problem. Some
dataset is stored in csv file and some dataset is stored in hdf5 file etc. Due to lacks of 
consistency of file type and data structure. The data preprocessing process is hard to decouple
and migration.  

The issues mentioned above reflect the complexity of the data science. This package gives a try on
dataset in modulating recognition field.  

In this project the two dataset is processed:
- [Deep sig's 2018.01A dataset](https://www.deepsig.ai/datasets/)
- [Hisar 2019](https://ieee-dataport.org/open-access/hisarmod-new-challenging-modulated-signals-dataset)
## usage
To ensure load the data correctly, check the path in the config file. Feel free to change the config
settings in config file.
The default code works on projects below.  
- dataset
  - 2019
    - HisarMod2019.1
      - Train
      - Test  
    
Use the command as below and follow the instruction, the original dataset will be
processed. 
```commandline
python loader.py
```
Follow the instruction
After the process finished successfully, the integrated data is stored in processed dir.
- processed
  - train
    - data0.csv
    - ...
  - test
    - data0.csv
    - ...
## features
1. load and store data in small batches
2. preprocess the data

## reference
1. rich
2. regular expression
3. pandas
4. h5py



