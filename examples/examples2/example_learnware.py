import joblib
import numpy as np
from sklearn import svm
from svm import SVM

from learnware.learnware import Learnware
import learnware.specification as specification
from learnware.utils import get_module_by_module_path


def prepare_learnware():
    data_X = np.random.randn(5000, 20)
    data_y = np.random.randn(5000)
    data_y = np.where(data_y > 0, 1, 0)

    clf = svm.SVC()
    clf.fit(data_X, data_y)
    joblib.dump(clf, "./svm/svm.pkl")

    spec = specification.utils.generate_rkme_spec(X=data_X, gamma=0.1)

    spec.save("./svm/spec.json")


def test_API():
    text_X = np.random.randn(100, 20)
    svm = SVM()
    pred_y1 = svm.predict(text_X)
    print(type(svm))

    model = {"module_path": "./svm/__init__.py", "class_name": "SVM"}
    spec = specification.rkme.RKMEStatSpecification()
    spec.load("./svm/spec.json")
    learnware = Learnware(id="A0", name="SVM", model=model, specification=spec, desc="svm")
    pred_y2 = learnware.predict(text_X)
    print(type(learnware.model))
    print(f"diff: {np.sum(pred_y1 != pred_y2)}")


if __name__ == "__main__":
    prepare_learnware()
    test_API()
