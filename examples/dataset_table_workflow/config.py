from learnware.tests.benchmarks import BenchmarkConfig


homo_n_labeled_list = [100, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000]
homo_n_repeat_list = [10, 10, 10, 3, 3, 3, 3, 3, 3]
hetero_n_labeled_list = [10, 30, 50, 75, 100, 200]
hetero_n_repeat_list = [10, 10, 10, 10, 10, 10]


user_semantic = {
    "Data": {"Values": ["Table"], "Type": "Class"},
    "Task": {"Values": ["Regression"], "Type": "Class"},
    "Library": {"Values": ["Others"], "Type": "Class"},
    "Scenario": {"Values": ["Business"], "Type": "Tag"},
    "Description": {"Values": "", "Type": "String"},
    "Name": {"Values": "", "Type": "String"},
}

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
    "lr": 1e-4, 
    "num_epoch": 50,
    "batch_size": 64,  
    "num_partition": 2,  # num of column partitions for pos/neg sampling
    "overlap_ratio": 0.7,  # specify the overlap ratio of column partitions during the CL
    "hidden_dim": 256,  # the dimension of hidden embeddings
    "num_layer": 6,  # the number of transformer layers used in the encoder
    "num_attention_head": 8,  # the numebr of heads of multihead self-attention layer in the transformers, should be divisible by hidden_dim
    "hidden_dropout_prob": 0.5,  # the dropout ratio in the transformer encoder
    "ffn_dim": 512,  # the dimension of feed-forward layer in the transformer layer
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
    },
    "M5": {
        "lgb": {
            "params": {
                "boosting_type": "gbdt",
                "objective": "rmse",
                "metric": "rmse",
                "learning_rate": 0.015,
                "num_leaves": 300,
                "max_depth": 500,
                "n_estimators": 100000,
                "boost_from_average": False,
                "num_threads": 32,
                "verbose": -1
            },
            "MAX_ROUNDS": 1000,
            "early_stopping_rounds": 1000
        }
    }
}

homo_table_benchmark_config = BenchmarkConfig(
    name="Corporacion",
    user_num=54,
    learnware_ids=[
        "00000912", "00000911", "00000910", "00000909",
        "00000908", "00000907", "00000906", "00000905",
        "00000904", "00000903", "00000902", "00000901",
        "00000900", "00000899", "00000898", "00000897",
        "00000896", "00000895", "00000894", "00000893",
        "00000892", "00000891", "00000890", "00000889",
        "00000888", "00000887", "00000886", "00000885",
        "00000884", "00000883", "00000882", "00000881",
        "00000880", "00000879", "00000878", "00000877",
        "00000876", "00000875", "00000874", "00000873",
        "00000872", "00000871", "00000870", "00000869",
        "00000868", "00000867", "00000866", "00000865",
        "00000864", "00000863", "00000862", "00000861",
        "00000860", "00000859"
    ],
    test_data_path="Corporacion/test_data.zip",
    train_data_path="Corporacion/train_data.zip",
    extra_info_path="Corporacion/extra_info.zip"
)

hetero_cross_feat_eng_benchmark_config = BenchmarkConfig(  
    name="PFS",
    user_num=41,
    learnware_ids = [
        
    ],
    test_data_path=None,
    train_data_path=None,
    extra_info_path=None
)

hetero_cross_task_benchmark_config = BenchmarkConfig(
    name="M5",
    user_num=9,
    learnware_ids = [
        "00000394", "00000393", "00000392", "00000391",
        "00000390", "00000389", "00000388", "00000387",
        "00000386", "00000385", "00000384", "00000383",
        "00000382", "00000381", "00000380", "00000379",
        "00000378", "00000377", "00000376", "00000375",
        "00000374", "00000373", "00000372", "00000371",
        "00000370", "00000369", "00000368", "00000367",
        "00000366", "00000365", "00000364", "00000363",
        "00000362", "00000361", "00000360", "00000359",
        "00000358", "00000357", "00000356", "00000355",
        "00000354", "00000353", "00000352", "00000351",
        "00000350", "00000349", "00000348", "00000347",
        "00000346", "00000345", "00000344", "00000343",
        "00000342", "00000444", "00000443", "00000442",
        "00000441", "00000440", "00000439", "00000438",
        "00000437", "00000436", "00000435", "00000434",
        "00000433", "00000432", "00000431", "00000430",
        "00000429", "00000428", "00000427", "00000426",
        "00000425", "00000424", "00000423", "00000422",
        "00000421", "00000420", "00000419", "00000418",
        "00000417", "00000416", "00000415", "00000414",
        "00000413", "00000412", "00000411", "00000410",
        "00000409", "00000408", "00000407", "00000406",
        "00000405", "00000404", "00000403", "00000402",
        "00000401", "00000400", "00000399", "00000398",
        "00000397", "00000396", "00000395", "00000783",
        "00000782", "00000781", "00000780", "00000779",
        "00000778", "00000777", "00000776", "00000775",
        "00000774", "00000773", "00000772", "00000771",
        "00000770", "00000769", "00000768", "00000767",
        "00000766", "00000765", "00000764", "00000763",
        "00000762", "00000761", "00000760", "00000759",
        "00000758", "00000757", "00000756", "00000755",
        "00000754", "00000753", "00000752", "00000751",
        "00000750", "00000749", "00000748", "00000747",
        "00000746", "00000745", "00000744", "00000743",
        "00000742", "00000741", "00000740", "00000739",
        "00000738", "00000737", "00000736", "00000735",
        "00000734", "00000733", "00000732", "00000731",
        "00000730", "00000839", "00000838", "00000837",
        "00000836", "00000835", "00000834", "00000833",
        "00000832", "00000831", "00000830", "00000829",
        "00000828", "00000827", "00000826", "00000825",
        "00000824", "00000823", "00000822", "00000821",
        "00000820", "00000819", "00000818", "00000817",
        "00000816", "00000815", "00000814", "00000813",
        "00000812", "00000811", "00000810", "00000809",
        "00000808", "00000807", "00000806", "00000805",
        "00000804", "00000803", "00000802", "00000801",
        "00000800", "00000799", "00000798", "00000797",
        "00000796", "00000795", "00000794", "00000793",
        "00000792", "00000791", "00000790", "00000789",
        "00000788", "00000787", "00000786", "00000912",
        "00000911", "00000910", "00000909", "00000908",
        "00000907", "00000906", "00000905", "00000904",
        "00000903", "00000902", "00000901", "00000900",
        "00000899", "00000898", "00000897", "00000896",
        "00000895", "00000894", "00000893", "00000892",
        "00000891", "00000890", "00000889", "00000888",
        "00000887", "00000886", "00000885", "00000884",
        "00000883", "00000882", "00000881", "00000880",
        "00000879", "00000878", "00000877", "00000876",
        "00000875", "00000874", "00000873", "00000872",
        "00000871", "00000870", "00000869", "00000868",
        "00000867", "00000866", "00000865", "00000864",
        "00000863", "00000862", "00000861", "00000860",
        "00000859"
    ],
    test_data_path="M5/test_data.zip",
    train_data_path="M5/train_data.zip",
    extra_info_path="M5/extra_info.zip"
)