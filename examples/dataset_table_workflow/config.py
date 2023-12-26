from learnware.tests.benchmarks import BenchmarkConfig


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
    'multiple_aug': "Multiple Learnware Reuse (FeatAug)",
    'ensemble_pruning': "Multiple Learnware Reuse (EnsemblePrune)",
    'multiple_avg': "Multiple Learnware Reuse (Averaging)"
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

user_model_params = {
    "Corporacion": {
        "lgb": {
            "params": {
                "num_leaves": 31,
                "objective": "regression",
                "learning_rate": 0.1,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 2,
                "metric": "l2",
                "num_threads": 4,
                "verbose": -1,
            },
            "MAX_ROUNDS": 500,
            "early_stopping_rounds": 50,
        }
    }
}

homo_table_benchmark_config = BenchmarkConfig(
    name="Corporacion",
    user_num=54,
    learnware_ids=[
        "00000912",
        "00000911",
        "00000910",
        "00000909",
        "00000908",
        "00000907",
        "00000906",
        "00000905",
        "00000904",
        "00000903",
        "00000902",
        "00000901",
        "00000900",
        "00000899",
        "00000898",
        "00000897",
        "00000896",
        "00000895",
        "00000894",
        "00000893",
        "00000892",
        "00000891",
        "00000890",
        "00000889",
        "00000888",
        "00000887",
        "00000886",
        "00000885",
        "00000884",
        "00000883",
        "00000882",
        "00000881",
        "00000880",
        "00000879",
        "00000878",
        "00000877",
        "00000876",
        "00000875",
        "00000874",
        "00000873",
        "00000872",
        "00000871",
        "00000870",
        "00000869",
        "00000868",
        "00000867",
        "00000866",
        "00000865",
        "00000864",
        "00000863",
        "00000862",
        "00000861",
        "00000860",
        "00000859"
    ],
    test_data_path="Corporacion/test_data.zip",
    train_data_path="Corporacion/train_data.zip",
    extra_info_path="Corporacion/extra_info.zip",
)