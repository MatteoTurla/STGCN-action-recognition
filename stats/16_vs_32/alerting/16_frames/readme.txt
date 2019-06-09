Dataset: /home/Dataset/aggregorio_skeletons_numpy/alerting/train
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
Numero di frame: 16
Downsample: 2
Balance: True
Padding: False
removed: 0
modality: random_center
randome move: True
Numero Azioni 400
A05 	 50
A06 	 50
A07 	 50
A08 	 50
A09 	 50
A10 	 50
A11 	 50
A12 	 50
Dataset: /home/Dataset/aggregorio_skeletons_numpy/alerting/test
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
Numero di frame: 16
Downsample: 2
Balance: False
Padding: False
removed: 0
modality: center
randome move: False
Numero Azioni 130
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
numero epoche: 500
best model at epoch: 155
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
