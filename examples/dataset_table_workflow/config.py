n_labeled_list = [100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000]
n_repeat_list = [10, 10, 10, 3, 3, 3, 3, 3, 3]

styles = {
    'user_model': {"color": "navy", "marker": "o", "linestyle": "-"},  
    'select_score': {'color': 'gold', 'marker': 's', 'linestyle': '--'},  
    'oracle_score': {'color': 'darkorange', 'marker': '^', 'linestyle': '-.'},  
    'mean_score': {'color': 'gray', 'marker': 'x', 'linestyle': ':'},
    'single_aug': {'color': 'gold', 'marker': 's', 'linestyle': '--'},
    'multiple_avg': {'color': 'blue', 'marker': '*', 'linestyle': '-'},  
    'multiple_aug': {'color': 'purple', 'marker': 'd', 'linestyle': '--'},  
    'ensemble_pruning': {"color": "magenta", "marker": "d", "linestyle": "-."} 
}

labels = {
    'user_model': "User Model",  
    'single_aug': "Single Learnware Reuse (Select)",
    "select_score": "Single Learnware Reuse (Select)",
    # "Single Learnware Reuse (Avg)",
    # "Single Learnware Reuse (Oracle)",
    'multiple_aug': "Multiple Learnware Reuse (FeatAug)",
    'ensemble_pruning': "Multiple Learnware Reuse (EnsemblePrune)",
    'multiple_avg': "Multiple Learnware Reuse (Averaging)"
}

output_description = {
    "Dimension": 1,
    "Description": {
        "0": "Product sales on the date.",
    },
}

user_semantic = {
    "Data": {"Values": ["Table"], "Type": "Class"},
    "Task": {"Values": ["Regression"], "Type": "Class"},
    "Library": {"Values": ["Others"], "Type": "Class"},
    "Scenario": {"Values": ["Business"], "Type": "Tag"},
    "Description": {"Values": "", "Type": "String"},
    "Name": {"Values": "", "Type": "String"},
    "Output": output_description,
}

align_model_params = {
    "network_type": "ArbitraryMapping",  # ["ArbitraryMapping", "BaseMapping", "BaseMapping_BN", "BaseMapping_Dropout"]
    "num_epoch": 50,
    "lr": 1e-5,
    "dropout_ratio": 0.2,
    "activation": "relu",
    "use_bn": True,
    "hidden_dims": [128, 256, 128, 256],
}

market_mapping_params = {
    "lr": 1e-4,  # [5e-5, 1e-4, 2e-4, 5e-4],
    "num_epoch": 50,
    "batch_size": 64,  # [64, 128, 256, 512, 1024],
    "num_partition": 2,  # [2, 3, 4], # num of column partitions for pos/neg sampling
    "overlap_ratio": 0.7,  # [0.1, 0.3, 0.5, 0.7], # specify the overlap ratio of column partitions during the CL
    "hidden_dim": 256,  # [64, 128, 256, 512, 768, 1024], # the dimension of hidden embeddings
    "num_layer": 6,  # [4, 6, 8, 10, 12, 14, 16, 20], # the number of transformer layers used in the encoder
    "num_attention_head": 8,  # [4, 8, 16], # the numebr of heads of multihead self-attention layer in the transformers, should be divisible by hidden_dim
    "hidden_dropout_prob": 0.5,  # [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], # the dropout ratio in the transformer encoder
    "ffn_dim": 512,  # [128, 256, 512, 768, 1024], # the dimension of feed-forward layer in the transformer layer
    "activation": "leakyrelu",
}
