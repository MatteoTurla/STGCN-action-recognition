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
Numero di frame: 16
Downsample: 2
Balance: True
Padding: False
removed: 0
modality: random_center
randome move: True
Numero Azioni 350
A13 	 50
A14 	 50
A15 	 50
A16 	 50
A17 	 50
A18 	 50
A19 	 50
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
Numero di frame: 16
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
['fcn.weight', 'fcn.bias']
batch size: 512
numero epoche: 500
best model at epoch: 180
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
