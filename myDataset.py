import torch
from torch.utils import data
import numpy as np

import os
import glob

from random import randint

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, root_dataset, n_frames, campionamento=1):
        self.root_dataset = root_dataset

        self.list_poses = []
        self.list_labels = []

        self.n_frames = n_frames
        self.campionamento = campionamento #stride frame consecutivi

        self.classes, self.class_to_idx, self.idx_to_class = self._find_classes(self.root_dataset)

        #crea lista video e label leggendo dalla directory root
        for target in self.class_to_idx.keys():
            d = os.path.join(self.root_dataset, target)
            files = glob.glob(os.path.join(d, "*.npy"))
            for file in files:
                self.list_poses.append(file)
                self.list_labels.append(self.class_to_idx[target])

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {i: classes[i] for in in range(len(classes))}
        return classes, class_to_idx, idx_to_class

    def _campionamento(self, pose, campionamento):
        depth = pose.shape[1]
        if(campionamento == 1):
            return pose
        if(depth // self.n_frames >= campionamento):
            index = torch.arange(0, depth, campionamento)
            video = torch.index_select(pose, 1, index)
            return pose
        else:
            return self._campionamento(pose, campionamento -1)

    def _center(self, pose):
        center = pose.shape[1] // 2
        frame = self.n_frames // 2
        return pose[:,center-frame:center+frame,:,:]

    def auto_pading(self, data_numpy, size, random_pad=False):
        C, T, V, M = data_numpy.shape
        if T < size:
            begin = random.randint(0, size - T) if random_pad else 0
            data_numpy_paded = np.zeros((C, size, V, M))
            data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
            return data_numpy_paded
        else:
            return data_numpy

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        pose = self.list_poses[index]
        pose_numpy = np.load(pose)
        pose_padded = self.auto_pading(pose_numpy, self.n_frames)
        X = torch.from_numpy(pose_padded)
        y = self.list_labels[index]
        X = self._campionamento(X, self.campionamento)
        X = self._center(X)

        return X, y

    def bincount(self):
        return zip(torch.bincount(torch.tensor(self.list_labels)), self.classes)

    def print(self):
        print("Dataset:", self.root_dataset)
        print("Numero Azioni", self.__len__())
        print("Classi:", self.classes)
        print("Classi to index:")
        for c in self.class_to_idx:
            print("Label:", c, "index:", self.class_to_idx[c])
        print("Numero di frame utilizzati", self.n_frames,)
        print("Campionamento:", self.campionamento)
        bins_labels = self.bincount()
        for bin, label in bins_labels:
            print("\tbin:", bin.item(), "\tlabel:", label)

if __name__ == '__main__':
    dataset = Dataset('Dataset/aggregorio_skeletons_numpy/basic/train', 16, campionamento=2)
    dataset.print()
    dataloader = data.DataLoader(
                            dataset,
                            batch_size=32,
                        	shuffle=True,
                        	pin_memory=True,
                            num_workers = 4
                        )
    for batch, (X,y) in enumerate(dataloader):
        print("Batch:", batch)
        print("\t", X.shape)
        print("\t", y)
