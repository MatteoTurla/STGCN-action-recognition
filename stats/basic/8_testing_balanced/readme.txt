Dataset: Dataset/aggregorio_skeletons_numpy_balanced_same/basic/train
Numero Azioni 340
Classi: ['A01', 'A02', 'A03', 'A04']
Classi to index:
Label: A01 index: 0
Label: A02 index: 1
Label: A03 index: 2
Label: A04 index: 3
Numero di frame utilizzati 8
Campionamento: 4
	bin: 85 	label: A01
	bin: 85 	label: A02
	bin: 85 	label: A03
	bin: 85 	label: A04
Dataset: Dataset/aggregorio_skeletons_numpy_balanced_same/basic/test
Numero Azioni 120
Classi: ['A01', 'A02', 'A03', 'A04']
Classi to index:
Label: A01 index: 0
Label: A02 index: 1
Label: A03 index: 2
Label: A04 index: 3
Numero di frame utilizzati 8
Campionamento: 4
	bin: 40 	label: A01
	bin: 30 	label: A02
	bin: 40 	label: A03
	bin: 10 	label: A04
['fcn.weight', 'fcn.bias']
batch size: 512
SGD (
Parameter Group 0
    dampening: 0
    initial_lr: 0.1
    lr: 0.1
    momentum: 0.9
    nesterov: False
    weight_decay: 0.5
)
CrossEntropyLoss()
{'milestones': [35, 70], 'gamma': 0.1, 'base_lrs': [0.1], 'last_epoch': -1}
