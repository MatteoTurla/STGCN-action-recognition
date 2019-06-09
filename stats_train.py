import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
import pdb

from i3dpt import I3D
from myDataset import Dataset


cluster = 'alerting'
nome_cartella = '32_frames_layer_7'
nframes = 32
       
weight_path = f'model/best_model_{cluster}/best_model.pt'
out_dir = f'stats_train/{cluster}/{nome_cartella}'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

conf_matrix_path = f'{out_dir}/confusion_matrix.jpg'
conf_matrix_confidence_path = f'{out_dir}/confusion_matrix_confidence.jpg'
readme_path = f'{out_dir}/readme.txt'

dataset_path = f'/home/Dataset/aggregorio_videos_pytorch_boxcrop/{cluster}/test'
dataset = Dataset(dataset_path, 32, 2, balance=False, padding=False, data_aug=False)
loader = data.DataLoader(
                        dataset,
                        batch_size=32,
                    	shuffle=False,
                    	pin_memory=True,
                        num_workers = 4
                    )
print('loading model')
num_classes = len(dataset.classes)
model = I3D(num_classes=num_classes, modality='rgb')
model.load_state_dict(torch.load(weight_path))
model.cuda()
model.eval()
print('Done..')

print('Calculating Stats')
conf_matrix = torch.zeros(num_classes, num_classes)
conf_matrix_confidence = torch.zeros(num_classes, num_classes)
mean_conf_matrix = 0.

with torch.no_grad():
    for X, y in loader:
        X = X.cuda()
        y = y.cuda()

        y_, _ = model(X)

        _, y_label_ = torch.max(y_, 1)
        
        for i in range(len(y)):
            conf_matrix[y[i].item(), y_label_[i].item()] += 1

        for i in range(len(y_)):
            conf_matrix_confidence[y[i].item()] += y_[i].cpu()

_bins = dataset.bincount()
bins = [_bin.item() for _bin in _bins]
for i in range(num_classes):
    for j in range(num_classes):
        conf_matrix[i][j] = conf_matrix[i][j]/bins[i]
        conf_matrix_confidence[i][j] = conf_matrix_confidence[i][j]/bins[i]

mean_tot = 0
for i in range(num_classes):
    mean_tot += conf_matrix[i][i].item()
mean_tot /= num_classes
print('Done..')

print('Saving result')
orig_stdout = sys.stdout
with open(readme_path, 'w+') as f:
    sys.stdout = f
    dataset.print()
    print('accuracy:', mean_tot)
sys.stdout = orig_stdout

fig = plt.figure()
plt.matshow(conf_matrix.numpy())
plt.title(f'{cluster} 32 frames accuracy', y=1.1)
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicated Label')
plt.savefig(conf_matrix_path)

fig = plt.figure()
plt.matshow(conf_matrix_confidence.numpy())
plt.title(f'{cluster} 32 frames confidence', y=1.1)
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicated Label')
plt.savefig(conf_matrix_confidence_path)



