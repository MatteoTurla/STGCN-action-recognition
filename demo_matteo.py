#!/usr/bin/env python
import os
import json

import numpy as np
import torch
import skvideo.io

import tools
import tools.utils as utils

from net.st_gcn import Model
class Demo():
    """
        Demo for Skeleton-based Action Recgnition
    """
    def start(self):
        #basic
        nc_basic = 4
        nc_alerting = 8
        nc_daily = 7

        print('loading model')
        model_args = {
          'in_channels': 3,
          'num_class': nc_basic,
          'edge_importance_weighting': True,
          'graph_args': {
            'layout': 'openpose',
            'strategy': 'spatial'
          }
        }
        model_weight = 'stats/basic/testing/best_model.pt'
        model = Model(model_args['in_channels'], model_args['num_class'],
                        model_args['graph_args'], model_args['edge_importance_weighting'])
        model.load_state_dict(torch.load(model_weight))
        model.cuda()

        video_path = 'demo/video2.avi'
        video_poses = 'demo/poses'

        output_jsonpack_path = 'demo/skeleton_pack.json'
        output_video_path = 'demo/output.avi'

        print("generating json skeleton")
        video = utils.video.get_video_frames(video_path)
        height, width, _ = video[0].shape
        video_info = utils.openpose.json_pack(
            video_poses, width, height)
        with open(output_jsonpack_path, 'w+') as outfile:
            json.dump(video_info, outfile)

        # parse skeleton data
        print("generating numpy")
        pose, _ = utils.video.video_info_parsing(video_info, num_person_out=1)
        data = torch.from_numpy(pose)
        data = data.unsqueeze(0)
        data = data.float().cuda()
        print(data.shape)

        label_name_path = './Dataset/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]

        # extract feature
        print('\nNetwork forwad...')
        model.eval()
        output, feature = model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        print("output", output.shape)
        print("feature", feature.shape)
        intensity = (feature*feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()
        label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)
        print('Prediction result: {}'.format(label_name[label]))
        print('Done.')


        # visualization
        print('\nVisualization...')
        label_sequence = output.sum(dim=2).argmax(dim=0)
        print(label_sequence)
        label_name_sequence = [[label_name[p] for p in l ]for l in label_sequence]
        edge = model.graph.edge
        images = utils.visualization.stgcn_visualize(
            pose, edge, intensity, video,label_name[label] , label_name_sequence, 1080)
        print('Done.')

        # save video
        print('\nSaving...')
        output_result_dir = 'demo'
        output_result_path = 'demo/output_stgcn.avi'
        if not os.path.exists(output_result_dir):
            os.makedirs(output_result_dir)
        writer = skvideo.io.FFmpegWriter(output_result_path,
                                        outputdict={'-b': '300000000'})
        for img in images:
            writer.writeFrame(img)
        writer.close()
        print('The Demo result has been saved in {}.'.format(output_result_path))

if __name__ == '__main__':
    Demo().start()
