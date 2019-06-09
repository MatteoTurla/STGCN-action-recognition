Dataset: /home/Dataset/aggregorio_skeletons_numpy/basic/train
Classes: ['A01', 'A02', 'A03', 'A04']
Classes to index:
Label: A01 index: 0
Label: A02 index: 1
Label: A03 index: 2
Label: A04 index: 3
Numero di frame: 32
Downsample: 2
Balance: True
Padding: False
removed: 59
modality: random_center
randome move: True
Numero Azioni 276
A01 	 69
A02 	 69
A03 	 69
A04 	 69
Dataset: /home/Dataset/aggregorio_skeletons_numpy/basic/test
Classes: ['A01', 'A02', 'A03', 'A04']
Classes to index:
Label: A01 index: 0
Label: A02 index: 1
Label: A03 index: 2
Label: A04 index: 3
Numero di frame: 32
Downsample: 2
Balance: False
Padding: False
removed: 28
modality: center
randome move: False
Numero Azioni 92
A01 	 30
A02 	 20
A03 	 32
A04 	 10
['st_gcn_networks.8.gcn.conv.weight', 'st_gcn_networks.8.gcn.conv.bias', 'st_gcn_networks.8.tcn.0.weight', 'st_gcn_networks.8.tcn.0.bias', 'st_gcn_networks.8.tcn.2.weight', 'st_gcn_networks.8.tcn.2.bias', 'st_gcn_networks.8.tcn.3.weight', 'st_gcn_networks.8.tcn.3.bias', 'st_gcn_networks.9.gcn.conv.weight', 'st_gcn_networks.9.gcn.conv.bias', 'st_gcn_networks.9.tcn.0.weight', 'st_gcn_networks.9.tcn.0.bias', 'st_gcn_networks.9.tcn.2.weight', 'st_gcn_networks.9.tcn.2.bias', 'st_gcn_networks.9.tcn.3.weight', 'st_gcn_networks.9.tcn.3.bias', 'fcn.weight', 'fcn.bias']
batch size: 512
numero epoche: 500
best model at epoch: 201
SGD (
Parameter Group 0
    dampening: 0
    initial_lr: 0.001
    lr: 0.0001
    momentum: 0.9
    nesterov: False
    weight_decay: 0.5
)
CrossEntropyLoss()
{'milestones': [400], 'gamma': 0.1, 'base_lrs': [0.001], 'last_epoch': 499}
