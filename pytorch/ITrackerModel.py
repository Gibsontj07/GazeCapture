""" ===================================================================================================================
Pytorch model definition for a modified iTracker where estimated head pose values are added as a new input..

Author: Petr Kellnhofer (pkellnho@gmail.com), 2018.

Modified: Thomas Gibson (tjg1g19@soton.ac.uk), 2022.

Website: http://gazecapture.csail.mit.edu/

Original Paper: Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

Date: 05/04/2022
====================================================================================================================="""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class FaceImageModel(nn.Module):

    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = ItrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize=25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ITrackerModel(nn.Module):

    def __init__(self):
        super(ITrackerModel, self).__init__()
        self.eyeModel = ItrackerImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel()
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2 * 12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
        )
        # Joining everything
        # Added capacity for head pose values (+2) and increased output size by the same amount
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 128 + 2, 130),
            nn.ReLU(inplace=True),
            nn.Linear(130, 2),
        )

    def forward(self, faces, eyesLeft, eyesRight, faceGrids, poses):
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)

        # Face net
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        # Insert head pose values into final fully connected layer by creating a flat vector with all inputs
        x = torch.cat((xEyes, xFace, xGrid, poses), 1)
        x = self.fc(x)

        return x
