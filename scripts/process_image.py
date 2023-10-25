#!/usr/bin/env python
# coding: utf-8

import os,sys
sys.path.insert(0,"..")
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage, skimage.io
import pprint

import torch
import torch.nn.functional as F
import torchvision, torchvision.transforms
import json

import torchxrayvision as xrv
import warnings
from io import StringIO

warnings.filterwarnings("ignore", message="Warning: Input size .* is not the native resolution .* for this model. A resize will be performed but this could impact performance.")

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('img_path', type=str)
parser.add_argument('-weights', type=str,default="densenet121-res224-all")
parser.add_argument('-feats', default=False, help='', action='store_true')
parser.add_argument('-cuda', default=False, help='', action='store_true')
parser.add_argument('-resize', default=False, help='', action='store_true')

cfg = parser.parse_args()

# Save the original stdout
original_stdout = sys.stdout

# Redirect stdout to a StringIO object
sys.stdout = StringIO()

img = skimage.io.imread(cfg.img_path)
img = xrv.datasets.normalize(img, 255)  

# Check that images are 2D arrays
if len(img.shape) > 2:
    img = img[:, :, 0]
if len(img.shape) < 2:
    print("error, dimension lower than 2 for image")

# Add color channel
img = img[None, :, :]


# the models will resize the input to the correct size so this is optional.
if cfg.resize:
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
else:
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

img = transform(img)


model = xrv.models.get_model(cfg.weights)

output = {}
with torch.no_grad():
    img = torch.from_numpy(img).unsqueeze(0)
    if cfg.cuda:
        img = img.cuda()
        model = model.cuda()
        
    if cfg.feats:
        feats = model.features(img)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        output["feats"] = list(feats.cpu().detach().numpy().reshape(-1))

    preds = model(img).cpu()
    output["preds"] = dict(zip(xrv.datasets.default_pathologies, map(float, preds[0].detach().numpy())))
    
if cfg.feats:
    print(output)
else:
    pprint.pprint(output)

results = {'preds': {...}}

results_json = json.dumps(output, indent=4)

with open('analysis_results.json', 'w') as f:
    f.write(results_json)

# Get the content of stdout
stdout_content = sys.stdout.getvalue()

# Restore the original stdout
sys.stdout = original_stdout

# Filter out the warning message
filtered_content = stdout_content.replace("Warning: Input size (222x222) is not the native resolution (224x224) for this model. A resize will be performed but this could impact performance.\n", "")

# Print the remaining content
print(results_json)

    
   
