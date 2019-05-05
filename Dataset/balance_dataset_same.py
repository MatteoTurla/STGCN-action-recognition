import os
import glob
import json
import torch
from random import randint
"""
Bilancia train set -> tutte le azioni hanno lo stesso numero di video

input -> dir path of root datatset
"""
class BalancedDatasetSame:
    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.max = 0
        debug = False

        print(self.root_dir)
        actions_cluster_paths = [d.path for d in os.scandir(self.root_dir) if d.is_dir]
        for acp in actions_cluster_paths:
            print("\t", acp)
            #voglio bilanciare solo il  train con grandezza pari al max in cluster
            max = 0
            acp = os.path.join(acp, 'train')
            actions_paths = [d.path for d in os.scandir(acp) if d.is_dir]
            for ap in actions_paths:
                print("\t\t scannning", ap)
                n_videos = self._n_files_in_action_folder(ap)
                if n_videos > max:
                    max = n_videos
            print("\t\t\t max:", max)
            print()
            #qui inizia il bilanciamento all'interno di un cluster {basic, alerting, daily_life}
            for ap in actions_paths:
                print("\t\t balancing", ap)
                n_videos = self._n_files_in_action_folder(ap)
                replicate = max // n_videos - 1
                random_i = max % n_videos
                print("\t\t\t", n_videos, replicate, random_i)
                files_paths = [f.path for f in os.scandir(ap)]
                for rep in range(replicate):
                    for i in range(len(files_paths)):
                        filep = files_paths[i]
                        file_name = filep.split('.')[0]
                        dist_pt_file = file_name+"_replicate_"+str(rep)+"_"+str(i)+".npy"
                        save = dist_pt_file
                        cmd = "cp {} {}".format(filep, save)
                        if not debug:
                            os.system(cmd)
                        print(cmd)
                for rand_i in range(random_i):
                    index = randint(0, len(files_paths)-1)
                    filep = files_paths[index]
                    file_name = filep.split('.')[0]
                    dist_pt_file = file_name+"_random_"+str(rand_i)+".npy"
                    save = dist_pt_file
                    cmd = "cp {} {}".format(filep, save)
                    if not debug:
                        os.system(cmd)
                    print(cmd)


    def _n_files_in_action_folder(self, folder):
        files_paths = [f.path for f in os.scandir(folder)]
        return len(files_paths)




if __name__ == '__main__':
    BalancedDatasetSame('aggregorio_skeletons_numpy_balanced_same')
