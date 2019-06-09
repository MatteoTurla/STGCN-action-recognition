Dataset: /home/Dataset/aggregorio_skeletons_numpy/basic/train
Classes: ['A01', 'A02', 'A03', 'A04']
Classes to index:
Label: A01 index: 0
Label: A02 index: 1
Label: A03 index: 2
Label: A04 index: 3
Numero di frame: 16
Downsample: 2
Balance: True
Padding: False
removed: 5
modality: random_center
randome move: True
Numero Azioni 340
A01 	 85
A02 	 85
A03 	 85
A04 	 85
Dataset: /home/Dataset/aggregorio_skeletons_numpy/basic/test
Classes: ['A01', 'A02', 'A03', 'A04']
Classes to index:
Label: A01 index: 0
Label: A02 index: 1
Label: A03 index: 2
Label: A04 index: 3
Numero di frame: 16
Downsample: 2
Balance: False
Padding: False
removed: 1
modality: center
randome move: False
Numero Azioni 119
A01 	 40
A02 	 29
A03 	 40
A04 	 10
['fcn.weight', 'fcn.bias']
batch size: 512
numero epoche: 500
best model at epoch: 340
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
