import torch
from torch.utils import data
import numpy as np
import queue

import os
import glob

from random import randint
import random

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, root_dataset, n_frames, campionamento=1, balance=True, padding=True, modality='center', move=True):
        self.root_dataset = root_dataset

        self.list_poses = []
        self.list_labels = []

        self.n_frames = n_frames
        self.campionamento = campionamento #stride frame consecutivi
        self.balance = balance
        self.padding = padding
        self.removed = 0
        self.modality = modality
        self.move = move
        if self.modality != 'center' and self.modality != 'random_center':
            raise Exception('modality errata')

        self.classes, self.class_to_idx, self.idx_to_class = self._find_classes(self.root_dataset)

        #crea lista video e label leggendo dalla directory root
        for target in self.class_to_idx.keys():
            d = os.path.join(self.root_dataset, target)
            files = glob.glob(os.path.join(d, "*.npy"))
            for file in files:
                self.list_poses.append(file)
                self.list_labels.append(self.class_to_idx[target])
        
        if not padding:
            self.list_poses, self.list_labels = self._removeUnfeasible()    

        if balance:
            self._balance()

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {i: classes[i] for i in range(len(classes))}
        return classes, class_to_idx, idx_to_class

    def _balance(self):
        print('Dataset -> balancing')
        bins = self.bincount()
        _max = bins.max().item()
        for i in range(len(bins)):
            _bin = bins[i].item()
            replicate = _max // _bin - 1
            _random = _max % _bin

            if i > 0:
                    start = bins[:i].sum().item()
            else: 
                start = 0

            for rep in range(replicate):
                for k in range(_bin):
                    self.list_poses.append(self.list_poses[k+start])
                    self.list_labels.append(self.list_labels[k+start])
            for _rand in range(_random):
                j = randint(start, start+_bin - 1)
                self.list_poses.append(self.list_poses[j])
                self.list_labels.append(self.list_labels[j])

        #sanity check
        bins = self.bincount()
        for i in range(1, len(bins)):
            if bins[i].item() != bins[i-1].item():
                raise Exception('Balancing gone wrong!')
        print('Done..')

    def _removeUnfeasible(self):
        print('Dataset -> searching for unfeasible')
        toDel = queue.Queue()
        for i in range(len(self.list_poses)):
            video = np.load(self.list_poses[i])
            if video.shape[1] < self.n_frames:
                toDel.put(i)
        print('Dataset -> deleting unfeasible')
        list_poses_tmp = []
        list_labels_tmp = []
        print(toDel.qsize())
        todel = -1
        if not toDel.empty():
            todel = toDel.get()
        for i in range(len(self.list_poses)):
            if i == todel:
                self.removed += 1
                if not toDel.empty():
                    todel = toDel.get()
            else:
                list_poses_tmp.append(self.list_poses[i])
                list_labels_tmp.append(self.list_labels[i])
        print('Done..')
        return list_poses_tmp, list_labels_tmp

    def _campionamento(self, pose, campionamento):
        depth = pose.shape[1]
        if(campionamento == 1):
            return pose
        if(depth // self.n_frames >= campionamento):
            return pose[:,::campionamento,:,:]
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

    def random_move(self,data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
        # input: C,T,V,M
        C, T, V, M = data_numpy.shape
        move_time = random.choice(move_time_candidate)
        node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
        node = np.append(node, T)
        num_node = len(node)

        A = np.random.choice(angle_candidate, num_node)
        S = np.random.choice(scale_candidate, num_node)
        T_x = np.random.choice(transform_candidate, num_node)
        T_y = np.random.choice(transform_candidate, num_node)

        a = np.zeros(T)
        s = np.zeros(T)
        t_x = np.zeros(T)
        t_y = np.zeros(T)

        # linspace
        for i in range(num_node - 1):
            a[node[i]:node[i + 1]] = np.linspace(
                A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
            s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                                 node[i + 1] - node[i])
            t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                                   node[i + 1] - node[i])
            t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                                   node[i + 1] - node[i])

        theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                          [np.sin(a) * s, np.cos(a) * s]])

        # perform transformation
        for i_frame in range(T):
            xy = data_numpy[0:2, i_frame, :, :]
            new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]
            data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

        return data_numpy

    def _random_normal(self, video):
        mean = video.shape[1] // 2
        range_frames = self.n_frames // 2
        valid_center = False

        while not(valid_center):
            norm_center_index = int(np.random.normal(loc=mean, scale=3.0)) #invce che 5.0
            if norm_center_index-range_frames >= 0 and norm_center_index+range_frames <= video.shape[1]:
                 valid_center = True

        return video[:,norm_center_index-range_frames:norm_center_index+range_frames,:,:]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        pose = self.list_poses[index]
        pose_numpy = np.load(pose)
        if self.padding:
            pose_numpy = self.auto_pading(pose_numpy, self.n_frames)
        if self.move:
            pose_numpy = self.random_move(pose_numpy)
        X = torch.from_numpy(pose_numpy)
        y = self.list_labels[index]
        X = self._campionamento(X, self.campionamento)
        if self.modality == 'center':
            X = self._center(X)
        elif self.modality == 'random_center':
            X = self._random_normal(X)
        return X, y

    def bincount(self):
        return torch.bincount(torch.tensor(self.list_labels))

    def print(self):
        print("Dataset:", self.root_dataset)
        print("Classes:", self.classes)
        print("Classes to index:")
        for c in self.class_to_idx:
            print("Label:", c, "index:", self.class_to_idx[c])
        print("Numero di frame:", self.n_frames,)
        print("Downsample:", self.campionamento)
        print("Balance:", self.balance)
        print("Padding:", self.padding)
        print("removed:", self.removed)
        print("modality:", self.modality)
        print("randome move:", self.move)
        print("Numero Azioni", self.__len__())
        bins = self.bincount()
        for idx, _bin in enumerate(bins):
            print(self.idx_to_class[idx], "\t", _bin.item())

if __name__ == '__main__':
    dataset = Dataset('/home/Dataset/aggregorio_skeletons_numpy/basic/train', 32, campionamento=1,
                     padding=False, balance=True, modality='center')
    dataset.print()
    for x, y in dataset:
        print(x.shape)
   
