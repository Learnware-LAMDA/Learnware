import os
import random
from time import sleep
from typing import Dict

import learnware
import numpy as np
import torch.random
from learnware import specification
from learnware.market import BaseUserInfo
from tqdm import tqdm

from build_market import user_semantic
from preprocess.dataloader import ImageDataLoader
from utils.clerk import Clerk, get_custom_logger
from utils.reuse import AveragingReuser


def evaluate_market_performance(args, market, clerk: Clerk=None, regenerate=True) -> Dict:
    logger = get_custom_logger()

    data_root = os.path.join(args.data_root, 'learnware_market_data', "{}_{:d}".format(args.data, args.data_id))
    dataloader = ImageDataLoader(data_root, args.n_users, train=False)
    acc = []

    market_root = args.market_root
    # shuffled = list(enumerate(dataloader))
    # random.shuffle(shuffled)
    for i, (test_X, test_y) in enumerate(dataloader):
        dir_path = os.path.join(market_root, args.data, "{}_{:d}".format(args.spec, args.id), "user_{:d}".format(i))
        os.makedirs(dir_path, exist_ok=True)

        if regenerate:
            if args.spec == "rbf":
                stat_spec = specification.utils.generate_rkme_spec(X=test_X, reduced_set_size=args.K, gamma=0.1, cuda_idx=args.cuda_idx)
            elif args.spec == "ntk":
                stat_spec = learnware.specification.RKMEImageStatSpecification(rkme_id=i+args.n_uploaders, **args.__dict__)
                stat_spec.generate_stat_spec_from_data(test_X, reduce=True, steps=args.ntk_steps, K=args.K, whitening=False)
            else:
                raise NotImplementedError()
            # Save User's spec to disk
            stat_spec.save(os.path.join(dir_path, "spec.json"))
        else:
            if args.spec == "rbf":
                stat_spec = specification.RKMEStatSpecification(gamma=0.1, cuda_idx=args.cuda_idx)
            elif args.spec == "ntk":
                stat_spec = learnware.specification.RKMEImageStatSpecification(rkme_id=i+args.n_uploaders, cache=False, **args.__dict__)
            else:
                raise NotImplementedError()
            # Load User's spec from disk
            stat_spec.load(os.path.join(dir_path, "spec.json"))

        user_info = BaseUserInfo(semantic_spec=user_semantic, stat_info={"RKMEStatSpecification": stat_spec})

        sorted_score_list, single_learnware_list, _, _= market.search_learnware(user_info, max_search_num=args.max_search_num)

        reuse_ensemble = AveragingReuser(learnware_list=single_learnware_list, mode="vote")
        ensemble_predict_y = np.argmax(reuse_ensemble.predict(user_data=test_X), axis=-1)

        curr_acc = np.mean(ensemble_predict_y == test_y)
        acc.append(curr_acc)
        if clerk:
            clerk.rkme_performance(curr_acc)

        logger.debug("Accuracy for user {:d}: {:.3f}; {:.3f} on average up to now.".format(i, curr_acc, np.mean(acc)))

    logger.info("Accuracy {:.3f}({:.3f})".format(np.mean(acc), np.std(acc)))

    return {
        "Accuracy": {
            "Mean": np.mean(acc),
            "Std": np.std(acc),
            "All": acc
        }
    }