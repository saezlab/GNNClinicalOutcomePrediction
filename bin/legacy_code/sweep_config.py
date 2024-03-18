parameter_search_space = {
    "unit": ["month"],
    "label": ["OSMonth"] ,
    "model": ["GATConv"],
    "bs": [16 ],
    "lr": [0.01],
    "weight_decay": [0.001],
    "num_of_gcn_layers": [3], 
    "num_of_ff_layers": [1],
    "gcn_h": [64], 
    "fcl": [128],
    "dropout": [0.1],
    "aggregators": ["None"],
    "scalers": ["identity"],
    "heads" : [1],
    "epoch": [2],
    "en": ["test_experiment"],
    "full_training": [False],
    "fold": [False],
    "loss": [None]
}

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep_',
    'metric': {
        'goal': 'minimize',
        'name': 'fold_val_mse_score'
    },
    'parameters': {
        key: {'values': value} for key, value in parameter_search_space.items()
    }
}
