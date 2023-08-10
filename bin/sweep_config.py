parameter_search_space = {
    "unit": ["month"],
    "label": ["OSMonth"] ,
    "model": ["PNAConv"],
    "bs": [16],
    "lr": [0.01],
    "weight_decay": [0.001],
    "num_of_gcn_layers": [3], 
    "num_of_ff_layers": [1],
    "gcn_h": [32], 
    "fcl": [512],
    "dropout": [0.1],
    "aggregators": ["max"],
    "scalers": ["identity"],
    "heads" : [1],
    "epoch": [2],
    "en": ["test_experiment"],
    "full_training": [True],
    "fold": [True],
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
