from flask import *
import os
from helper import *
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import os
import copy

main = Blueprint('main', __name__, template_folder='templates')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

@main.route('/', methods=['GET', 'POST'])
def main_route():
	validFormats = set(['png', 'jpg', 'bmp', 'gif'])
	style_img = image_loader(request.files.get("picStyle"))
	content_img = image_loader(request.files.get("picContent"))

	return render_template("index.html")