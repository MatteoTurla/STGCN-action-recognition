Dataset: Dataset/aggregorio_skeletons_numpy_balanced_same/alerting/train
Numero Azioni 400
Classi: ['A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
Classi to index:
Label: A05 index: 0
Label: A06 index: 1
Label: A07 index: 2
Label: A08 index: 3
Label: A09 index: 4
Label: A10 index: 5
Label: A11 index: 6
Label: A12 index: 7
Numero di frame utilizzati 32
Campionamento: 2
	bin: 50 	label: A05
	bin: 50 	label: A06
	bin: 50 	label: A07
	bin: 50 	label: A08
	bin: 50 	label: A09
	bin: 50 	label: A10
	bin: 50 	label: A11
	bin: 50 	label: A12
Dataset: Dataset/aggregorio_skeletons_numpy_balanced_same/alerting/test
Numero Azioni 130
Classi: ['A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12']
Classi to index:
Label: A05 index: 0
Label: A06 index: 1
Label: A07 index: 2
Label: A08 index: 3
Label: A09 index: 4
Label: A10 index: 5
Label: A11 index: 6
Label: A12 index: 7
Numero di frame utilizzati 32
Campionamento: 2
	bin: 20 	label: A05
	bin: 20 	label: A06
	bin: 10 	label: A07
	bin: 20 	label: A08
	bin: 20 	label: A09
	bin: 10 	label: A10
	bin: 20 	label: A11
	bin: 10 	label: A12
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
