import os,sys
sys.path.insert(0, "..")
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import skimage
import skimage.io
import pprint

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms
import json

import torchxrayvision as xrv
import warnings
from io import StringIO

warnings.filterwarnings("ignore", message="Warning: Input size .* is not the native resolution .* for this model. A resize will be performed but this could impact performance.")

def process_image(img_path, weights="densenet121-res224-all", feats=False, cuda=False, resize=False):
    img = skimage.io.imread(img_path)
    img = xrv.datasets.normalize(img, 255)  

    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    # Add color channel
    img = img[None, :, :]

    # the models will resize the input to the correct size so this is optional.
    if resize:
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                    xrv.datasets.XRayResizer(224)])
    else:
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

    img = transform(img)

    model = xrv.models.get_model(weights)

    output = {}
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        if cuda:
            img = img.cuda()
            model = model.cuda()
        
        if feats:
            feats = model.features(img)
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            output["feats"] = list(feats.cpu().detach().numpy().reshape(-1))

        preds = model(img).cpu()
        output["preds"] = dict(zip(xrv.datasets.default_pathologies, map(float, preds[0].detach().numpy())))
    
    results_json = json.dumps(output, indent=4)
    return results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default="", help='')
    parser.add_argument('img_path', type=str)
    parser.add_argument('-weights', type=str, default="densenet121-res224-all")
    parser.add_argument('-feats', default=False, help='', action='store_true')
    parser.add_argument('-cuda', default=False, help='', action='store_true')
    parser.add_argument('-resize', default=False, help='', action='store_true')

    cfg = parser.parse_args()

    results_json = process_image(cfg.img_path, cfg.weights, cfg.feats, cfg.cuda, cfg.resize)
    
    with open('analysis_results.json', 'w') as f:
        f.write(results_json)
    
    print(results_json)
