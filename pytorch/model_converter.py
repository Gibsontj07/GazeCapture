""" ===================================================================================================================
Code to convert pytorch model saved in checkpoint.pth.tar format to onnx model.

Author: Thomas Gibson (tjg1g19@soton.ac.uk), 2022.

Date: 05/04/2022
====================================================================================================================="""

import os
import torch.onnx
from ITrackerModel import ITrackerModel
from pytorch.main import CHECKPOINTS_PATH


def load_checkpoint(filename='best_checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state


model = ITrackerModel()

# Load pretrained model weights
model_url = 'best_checkpoint.pth.tar'
batch_size = 1  # just a random number
saved = load_checkpoint()

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None

state = saved['state_dict']
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

try:
    model.Module.load_state_dict(new_state_dict)
except:
    model.load_state_dict(new_state_dict)

epoch = saved['epoch']
best_prec1 = saved['best_prec1']

# set the model to inference mode
model.eval()

# Input to the model
grid = torch.rand(batch_size, 625, requires_grad=True)
img = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
pose = torch.tensor([[0.5, 0.5]], requires_grad=True)
torch_out = model(img, img, img, grid, pose)

# Export the model
torch.onnx.export(model,  # model being run
                  (img, img, img, grid, pose),  # model input (or a tuple for multiple inputs)
                  "ORAS_Model.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  verbose=True,
                  opset_version=11,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['face', 'left', 'right', 'grid', 'pose'],  # the model's input names
                  output_names=['output'])  # the model's output names
