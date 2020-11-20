import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

TRAIN_PATH = '/home/vedank/Desktop/code/facial_sentiment/dataset/train.csv'
TEST_PATH = '/home/vedank/Desktop/code/facial_sentiment/dataset/test.csv'



def csv_to_np(PATH,dataset_type):
    a = []
    data = pd.read_csv(PATH)
    if dataset_type == "train":
        data,data_y = data.iloc[:,1:],data.iloc[:,:1]
        np.save("/home/vedank/Desktop/code/facial_sentiment/npy_data/train_data_labels.npy",np.array(data_y))
    data = pd.Series(np.array(data).flatten())
    for i in range(len(data)):  
        img_str = data[i].split(' ')
        image = np.asarray(img_str,dtype=np.uint8).reshape(48,48,1)
        a.append(image)

    data = np.array(a).squeeze(axis=3)

    data_tensor = torch.Tensor([np.repeat(image[...,np.newaxis],3,-1)/255.0 for image in data])
    np.save(f"/home/vedank/Desktop/code/facial_sentiment/npy_data/{dataset_type}_data.npy",np.array(data_tensor))


csv_to_np(TRAIN_PATH,"train")
csv_to_np(TEST_PATH,"test")
