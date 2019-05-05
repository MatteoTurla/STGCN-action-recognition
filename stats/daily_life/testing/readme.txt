Dataset: Dataset/aggregorio_skeletons_numpy/daily_life/train
Numero Azioni 250
Classi: ['A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']
Classi to index:
Label: A13 index: 0
Label: A14 index: 1
Label: A15 index: 2
Label: A16 index: 3
Label: A17 index: 4
Label: A18 index: 5
Label: A19 index: 6
Numero di frame utilizzati 32
Campionamento: 1
	bin: 35 	label: A13
	bin: 35 	label: A14
	bin: 35 	label: A15
	bin: 35 	label: A16
	bin: 50 	label: A17
	bin: 35 	label: A18
	bin: 25 	label: A19
Dataset: Dataset/aggregorio_skeletons_numpy/daily_life/test
Numero Azioni 130
Classi: ['A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']
Classi to index:
Label: A13 index: 0
Label: A14 index: 1
Label: A15 index: 2
Label: A16 index: 3
Label: A17 index: 4
Label: A18 index: 5
Label: A19 index: 6
Numero di frame utilizzati 32
Campionamento: 1
	bin: 20 	label: A13
	bin: 20 	label: A14
	bin: 20 	label: A15
	bin: 20 	label: A16
	bin: 20 	label: A17
	bin: 20 	label: A18
	bin: 10 	label: A19
['fcn.weight', 'fcn.bias']
batch size: 256
SGD (
Parameter Group 0
    dampening: 0
    initial_lr: 0.1
    lr: 0.1
    momentum: 0.9
    nesterov: False
    weight_decay: 0.01
)
CrossEntropyLoss()
{'milestones': [25, 45], 'gamma': 0.1, 'base_lrs': [0.1], 'last_epoch': -1}
