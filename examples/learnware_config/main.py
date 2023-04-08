from learnware.learnware import get_learnware_from_dirpath

import joblib
import numpy as np
import learnware.specification as specification

from sklearn import svm
from learnware.config import C
from learnware.learnware import get_learnware_from_dirpath

def prepare_learnware():
    data_X = np.random.randn(5000, 20)
    data_y = np.random.randn(5000)
    data_y = np.where(data_y > 0, 1, 0)

    clf = svm.SVC()
    clf.fit(data_X, data_y)
    joblib.dump(clf, "./svm/svm.pkl")

    spec = specification.utils.generate_rkme_spec(X=data_X, gamma=0.1, cuda_idx=0)
    spec.save("./svm/spec.json")


def test_import_learnware():
    learnware_inst = get_learnware_from_dirpath(id="123", semantic_spec=C.semantic_specs, learnware_dirpath="./svm")
    return learnware_inst

if __name__ == '__main__':
    prepare_learnware()
    test_import_learnware()