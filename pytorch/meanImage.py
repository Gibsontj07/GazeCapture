""" ===================================================================================================================
Code to create mean image from gazecapture dataset after preprocessing.

Author: Thomas Gibson (tjg1g19@soton.ac.uk), 2022.

Date: 05/04/2022
====================================================================================================================="""

import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import scipy.io as sio
import cv2
from ITrackerData import ITrackerData

parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--data_path', help="Path to processed dataset. It should contain metadata.mat. Use prepareDataset.py.")
args = parser.parse_args()

batch_size = 32
workers = 8

def main():
    global args
    imSize=(224,224)

    dataTrain = ITrackerData(dataPath = args.data_path, split='train', imSize = imSize)
    dataVal = ITrackerData(dataPath = args.data_path, split='test', imSize = imSize)

    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    mean_image = np.zeros((224, 224, 3))

    count = 0
    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(train_loader):
        mean_image, count = generate_mean_image(mean_image, imFace, count)
        print("batch ", count / 32, " processed")

    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(val_loader):
        mean_image, count = generate_mean_image(mean_image, imFace, count)
        print("batch ", count / 32, " processed")

    mean_image = mean_image[:, :, ::-1]

    mdic = {"image_mean": mean_image, "label": "image_mean"}
    sio.savemat("mean_face_224_dots.mat", mdic)

    cv2.imwrite('mean_face_dots.jpg', mean_image)


def generate_mean_image(mean_image, imFace, count):
    current_batch_size = imFace.shape[0]
    imFace = imFace.numpy()
    imFace = np.sum(imFace, axis=0)
    # Compute mean image
    if count == 0:
        mean_image = imFace / current_batch_size
        count += current_batch_size
    else:
        mean_image = mean_image * count + imFace
        count += current_batch_size
        mean_image = mean_image / count

    return mean_image, count


if __name__ == "__main__":
    main()
    print('DONE')
