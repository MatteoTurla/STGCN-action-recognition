Dataset: /home/Dataset/aggregorio_skeletons_numpy/daily_life/train
Classes: ['A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']
Classes to index:
Label: A13 index: 0
Label: A14 index: 1
Label: A15 index: 2
Label: A16 index: 3
Label: A17 index: 4
Label: A18 index: 5
Label: A19 index: 6
Numero di frame: 32
Downsample: 2
Balance: True
Padding: False
removed: 1
modality: random_center
randome move: True
Numero Azioni 343
A13 	 49
A14 	 49
A15 	 49
A16 	 49
A17 	 49
A18 	 49
A19 	 49
Dataset: /home/Dataset/aggregorio_skeletons_numpy/daily_life/test
Classes: ['A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']
Classes to index:
Label: A13 index: 0
Label: A14 index: 1
Label: A15 index: 2
Label: A16 index: 3
Label: A17 index: 4
Label: A18 index: 5
Label: A19 index: 6
Numero di frame: 32
Downsample: 2
Balance: False
Padding: False
removed: 0
modality: center
randome move: False
Numero Azioni 130
A13 	 20
A14 	 20
A15 	 20
A16 	 20
A17 	 20
A18 	 20
A19 	 10
['st_gcn_networks.8.gcn.conv.weight', 'st_gcn_networks.8.gcn.conv.bias', 'st_gcn_networks.8.tcn.0.weight', 'st_gcn_networks.8.tcn.0.bias', 'st_gcn_networks.8.tcn.2.weight', 'st_gcn_networks.8.tcn.2.bias', 'st_gcn_networks.8.tcn.3.weight', 'st_gcn_networks.8.tcn.3.bias', 'st_gcn_networks.9.gcn.conv.weight', 'st_gcn_networks.9.gcn.conv.bias', 'st_gcn_networks.9.tcn.0.weight', 'st_gcn_networks.9.tcn.0.bias', 'st_gcn_networks.9.tcn.2.weight', 'st_gcn_networks.9.tcn.2.bias', 'st_gcn_networks.9.tcn.3.weight', 'st_gcn_networks.9.tcn.3.bias', 'fcn.weight', 'fcn.bias']
batch size: 512
numero epoche: 500
best model at epoch: 190
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
