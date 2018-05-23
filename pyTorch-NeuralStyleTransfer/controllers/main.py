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





@main.route('/', methods=['GET', 'POST'])
def main_route():
	validFormats = set(['png', 'jpg', 'bmp', 'gif'])
	# if (request.form.get("op") == "img_search"):
	#   	picFile = request.files.get("pic")
	#   	if picFile:
	#   		filename = picFile.filename
	#   		extension = filename[-3:]
	#   		# make sure the file is of a valid picture format
	#   		if extension.lower() not in validFormats and filename[-4:].lower() not in ['tiff', 'jpeg']:
	#   			abort(404)
	#        	# save the uploaded image so we can send it to the model
	#        	curPath = os.path.dirname(__file__)
	#        	relPath = "static/images"
	#        	imagesFolder = os.path.abspath(os.path.join(curPath, os.pardir, relPath))
	#        	picFile.save(os.path.join(imagesFolder, filename))
	#        	# Send the file to the trained model
	#        	# model.recognize(filename)
	#        	# returns JSON data to search by
	#        	# store into set/list/just use the json object and grab the values for each key maybe
	#        		#call container imageInfo
	#        	#redirectURL = "http://www.wayfair.com/keyword.php?keyword="
	#        	redirectURL = "http://www.wayfair.com/keyword.php?keyword=blue+lamp"
	#        	# redirectURL += str(imageInfo[0]);
	#        	# for attr in imageInfo[1:]:
	#        	# 	redirectURL += ("+" + str(attr))
	#        	print redirectURL
	#        	return redirect(redirectURL)

	return render_template("index.html")