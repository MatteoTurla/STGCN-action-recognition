Dataset: ../Dataset/aggregorio_balanced/aggregorio_skeletons_numpy/alerting/train
Numero Azioni 400
Classes: ['A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
Classes to index:
Label: A05 index: 0
Label: A06 index: 1
Label: A07 index: 2
Label: A08 index: 3
Label: A09 index: 4
Label: A10 index: 5
Label: A11 index: 6
Label: A12 index: 7
Numero di frame: 32
Downsample: 1
A05 	 50
A06 	 50
A07 	 50
A08 	 50
A09 	 50
A10 	 50
A11 	 50
A12 	 50
Dataset: ../Dataset/aggregorio_balanced/aggregorio_skeletons_numpy/alerting/test
Numero Azioni 130
Classes: ['A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
Classes to index:
Label: A05 index: 0
Label: A06 index: 1
Label: A07 index: 2
Label: A08 index: 3
Label: A09 index: 4
Label: A10 index: 5
Label: A11 index: 6
Label: A12 index: 7
Numero di frame: 32
Downsample: 1
A05 	 20
A06 	 20
A07 	 10
A08 	 20
A09 	 20
A10 	 10
A11 	 20
A12 	 10
['fcn.weight', 'fcn.bias']
batch size: 512
numero epoche: 50
best model at epoch: 20
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
