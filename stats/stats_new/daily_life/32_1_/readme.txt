Dataset: ../Dataset/aggregorio_balanced/aggregorio_skeletons_numpy/daily_life/train
Numero Azioni 350
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
Downsample: 1
A13 	 50
A14 	 50
A15 	 50
A16 	 50
A17 	 50
A18 	 50
A19 	 50
Dataset: ../Dataset/aggregorio_balanced/aggregorio_skeletons_numpy/daily_life/test
Numero Azioni 130
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
Downsample: 1
A13 	 20
A14 	 20
A15 	 20
A16 	 20
A17 	 20
A18 	 20
A19 	 10
['fcn.weight', 'fcn.bias']
batch size: 512
numero epoche: 50
best model at epoch: 10
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
