import learnware
from pfs import Dataloader



if __name__ == "__main__":
    pfs = Dataloader()
    # pfs.regenerate_data()
    algo_list = ["ridge", "lgb"]
    for algo in algo_list:
        pfs.set_algo(algo)
        pfs.retrain_models()