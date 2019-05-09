Dataset: ../Dataset/aggregorio_balanced/aggregorio_skeletons_numpy/basic/train
Numero Azioni 340
Classes: ['A01', 'A02', 'A03', 'A04']
Classes to index:
Label: A01 index: 0
Label: A02 index: 1
Label: A03 index: 2
Label: A04 index: 3
Numero di frame: 32
Downsample: 1
A01 	 85
A02 	 85
A03 	 85
A04 	 85
Dataset: ../Dataset/aggregorio_balanced/aggregorio_skeletons_numpy/basic/test
Numero Azioni 120
Classes: ['A01', 'A02', 'A03', 'A04']
Classes to index:
Label: A01 index: 0
Label: A02 index: 1
Label: A03 index: 2
Label: A04 index: 3
Numero di frame: 32
Downsample: 1
A01 	 40
A02 	 30
A03 	 40
A04 	 10
['fcn.weight', 'fcn.bias']
batch size: 512
numero epoche: 50
best model at epoch: 18
SGD (
Parameter Group 0
    dampening: 0
    initial_lr: 0.5
    lr: 0.125
    momentum: 0.9
    nesterov: False
    weight_decay: 0.1
)
CrossEntropyLoss()
{'milestones': [15, 30], 'gamma': 0.5, 'base_lrs': [0.5], 'last_epoch': 49}
