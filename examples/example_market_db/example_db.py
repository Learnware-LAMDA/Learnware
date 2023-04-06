from learnware.market import EasyMarket
from learnware.market import database_ops
from learnware.learnware import Learnware
import learnware.specification as specification
from learnware.utils import get_module_by_module_path

from sklearn import svm
import joblib
import numpy as np
import os


def prepare_learnware(learnware_num = 10):
    for i in range(learnware_num):
        dir_path = f"./learnware_pool/svm{i}"
        os.makedirs(dir_path, exist_ok=True)
        
        print("Preparing Learnware: %d"%(i))
        data_X = np.random.randn(5000, 20)
        data_y = np.random.randn(5000)
        data_y = np.where(data_y > 0, 1, 0)

        clf = svm.SVC(kernel="linear")
        clf.fit(data_X, data_y)
        joblib.dump(clf, os.path.join(dir_path, "svm.pkl"))

        spec = specification.utils.generate_rkme_spec(X=data_X, gamma=0.1, cuda_idx=0)
        spec.save(os.path.join(dir_path, "spec.json"))
        
        init_file = os.path.join(dir_path, "__init__.py")
        os.system(f"cp example_init.py {init_file}")


def test_market():
    easy_market = EasyMarket()
    print('Total Item:', len(easy_market))
    test_learnware_num = 10
    prepare_learnware(test_learnware_num)
    root_path = "./learnware_pool"
    os.makedirs(root_path, exist_ok=True)
    
    for i in range(test_learnware_num):
        dir_path = f"./learnware_pool/svm{i}"
        model_path = os.path.join(dir_path, "__init__.py")
        stat_spec_path = os.path.join(dir_path, "spec.json")
        easy_market.add_learnware('learnware_%d'%(i), model_path, stat_spec_path, {"desc":"test_learnware_number_%d"%(i)})
    print('Total Item:', len(easy_market))
    curr_inds = easy_market._get_ids()
    print("Available ids:", curr_inds)

    easy_market.delete_learnware(curr_inds[4])
    easy_market.delete_learnware(curr_inds[8])
    curr_inds = easy_market._get_ids()
    print("Available ids:", curr_inds)


if __name__ == '__main__':
    test_market()