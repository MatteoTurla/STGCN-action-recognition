#!/usr/bin/env python
import os
import json

import numpy as np
import torch
import pdb
import skvideo.io

import tools
import tools.utils as utils

import docker
from pathlib import Path

from net.st_gcn import Model

class Demo():
    def start(self):
        cluster = 'basic'
        model_weight = 'stats/stats_new/{}/32_1_/best_model.pt'.format(cluster)
        demo_test_folder = 'demo/basic'

        openpose_folder_name = 'openpose'
        video_name = 'video.avi'

        openpose_folder_path = os.path.join(demo_test_folder, openpose_folder_name)
        if not os.path.exists(openpose_folder_path):
            os.makedirs(openpose_folder_path)
        
        docker_volume = Path(demo_test_folder).absolute()

        cmd = "./build/examples/openpose/openpose.bin --video ../data/{} --write_video \
                ../data/openpose.avi --write_keypoint_json  ../data/{} --no_display".format(video_name, openpose_folder_name)
        
        volumes = {docker_volume: {'bind': '/data', 'mode': 'rw'}}
        client = docker.from_env()
        print("extracting skeletons keypoints")
        result = client.containers.run("mjsobrep/openpose:latest", cmd, runtime="nvidia", volumes=volumes)
        print(result.decode('UTF-8'))

        # NON MODIFICARE  DA QUA IN POI 
        if cluster == 'basic':
            num_classes = 4
            offset = 0
        elif cluster == 'alerting':
            num_classes = 8
            offset = 4
        elif cluster == 'daily_life':
            num_classes = 7
            offset = 4+8

        print('loading model')
        model_args = {
          'in_channels': 3,
          'num_class': num_classes,
          'edge_importance_weighting': True,
          'graph_args': {
            'layout': 'openpose',
            'strategy': 'spatial'
          }
        }
        
        model = Model(model_args['in_channels'], model_args['num_class'],
                        model_args['graph_args'], model_args['edge_importance_weighting'])
        model.load_state_dict(torch.load(model_weight))
        model.cuda()

        video_path = '{}/{}'.format(demo_test_folder, video_name)
        video_poses = '{}/{}'.format(demo_test_folder, openpose_folder_name)
        output_result_path = '{}/stgcn.avi'.format(demo_test_folder)

        print("generating json skeleton")
        video = utils.video.get_video_frames(video_path)
        height, width, _ = video[0].shape
        video_info = utils.openpose.json_pack(
            video_poses, width, height)

        # parse skeleton data
        print("generating numpy")
        pose, _ = utils.video.video_info_parsing(video_info, num_person_out=1)
        data = torch.from_numpy(pose)
        data = data.unsqueeze(0)
        data = data.float().cuda()

        label_name_path = '../Dataset/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]

        # extract feature
        print('\nNetwork forwad...')
        model.eval()
        output, feature = model.extract_feature(data)
        output = output[0]
        feature = feature[0]

        intensity = (feature*feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()
        label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)
        print('Prediction result: {}'.format(label_name[label+offset]))
        print('Done.')


        # visualization
        print('\nVisualization...')
        label_sequence = output.sum(dim=2).argmax(dim=0)
        label_name_sequence = [[label_name[p+offset] for p in l ]for l in label_sequence]
        edge = model.graph.edge
        images = utils.visualization.stgcn_visualize_output(
            pose, edge, intensity, video,label_name[label+offset] , label_name_sequence, 1080)
        print('Done.')
        pdb.set_trace()

        # save video
        print('\nSaving...')
        writer = skvideo.io.FFmpegWriter(output_result_path,
                                        outputdict={'-b': '300000000'})
        for img in images:
            writer.writeFrame(img)
        writer.close()
        print('The Demo result has been saved in {}.'.format(output_result_path))

if __name__ == '__main__':  
    Demo().start()
    
    
