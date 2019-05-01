import os
from pathlib import Path
import json
import numpy as np

def json_pack(snippets_dir, frame_width=1, frame_height=1, label='unknown', label_index=-1):
    sequence_info = []
    p = Path(snippets_dir)
    for index, path in enumerate(sorted(p.glob('*.json'))):
        #print(path)
        json_path = str(path)
        #frame_id = int(path.stem.split('_')[-2])
        frame_data = {'frame_index': index}
        data = json.load(open(json_path))
        skeletons = []
        for person in data['people']:
            score, coordinates = [], []
            skeleton = {}
            keypoints = person['pose_keypoints']
            for i in range(0, len(keypoints), 3):
                coordinates += [keypoints[i]/frame_width, keypoints[i + 1]/frame_height]
                score += [keypoints[i + 2]]
            skeleton['pose'] = coordinates
            skeleton['score'] = score
            skeletons += [skeleton]
        frame_data['skeleton'] = skeletons
        sequence_info += [frame_data]

    video_info = dict()
    video_info['data'] = sequence_info
    video_info['label'] = label
    video_info['label_index'] = label_index

    return video_info

def video_info_parsing(video_info, num_person_in=5, num_person_out=1):
    data_numpy = np.zeros((3, len(video_info['data']), 18, num_person_in))
    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index']
        #print(frame_index)
        for m, skeleton_info in enumerate(frame_info["skeleton"]):
            if m >= num_person_in:
                break
            pose = skeleton_info['pose']
            score = skeleton_info['score']
            data_numpy[0, frame_index, :, m] = pose[0::2]
            data_numpy[1, frame_index, :, m] = pose[1::2]
            data_numpy[2, frame_index, :, m] = score

    # centralization
    data_numpy[0:2] = data_numpy[0:2] - 0.5
    data_numpy[0][data_numpy[2] == 0] = 0
    data_numpy[1][data_numpy[2] == 0] = 0

    sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
    for t, s in enumerate(sort_index):
        data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                    0))
    data_numpy = data_numpy[:, :, :, :num_person_out]

    return data_numpy

data_openpose = 'aggregorio_skeletons/skeletons_normalize'
dist_save = 'aggregorio_skeletons_numpy'
debug = False

print("root:",data_openpose)
print("dist:", dist_save)

for t in ['train', 'test']:
    t_path_root = os.path.join(data_openpose, t)
    t_path_dist = os.path.join(dist_save, t)
    if not debug:
        os.makedirs(t_path_dist)
    print("\t root:", t_path_root)
    print("\t dist:", t_path_dist)

    actions = [(d.path, d.name) for d in os.scandir(t_path_root) if d.is_dir]

    for (ap, an) in actions:
        a_path_root = ap
        a_path_dist = os.path.join(t_path_dist, an)
        if not debug:
            os.makedirs(a_path_dist)
        print("\t\t root:", a_path_root)
        print("\t\t dist:", a_path_dist)

        sequence_in_folderaction = [(d.path, d.name) for d in os.scandir(a_path_root) if d.is_dir]

        for (sp, sn) in sequence_in_folderaction:
            save_file_name = sn+".npy"
            s_path_root = sp
            s_path_dist = os.path.join(a_path_dist, save_file_name)
            print("\t\t\t root:", s_path_root)
            print("\t\t\t dist:", s_path_dist)
            json_pack_pose = json_pack(s_path_root)
            print("\t\t\t\t json:", len(json_pack_pose['data']))
            data_numpy = video_info_parsing(json_pack_pose)
            print("\t\t\t\t data:",data_numpy.shape)
            if not debug:
                np.save(s_path_dist, data_numpy)
