from learnware.learnware import Learnware
from learnware.reuse.base import BaseReuser
from .feature_alignment import FeatureAligner
from ..feature_augment_reuser import FeatureAugmentReuser


class HeteroMapTableReuser(BaseReuser):

    def __init__(self, learnware: Learnware = None, mode: str = None, cuda_idx=0, **align_arguments):
        self.learnware=learnware
        assert mode in ["classification", "regression"]
        self.mode=mode
        self.cuda_idx=cuda_idx
        self.align_arguments=align_arguments

    def fit(self, user_rkme):
        self.feature_aligner=FeatureAligner(learnware=self.learnware, mode=self.mode, cuda_idx=self.cuda_idx, **self.align_arguments)
        self.feature_aligner.fit(user_rkme)
        self.reuser=self.feature_aligner

    def finetune(self, x_train,y_train):
        self.reuser=FeatureAugmentReuser(learnware=self.feature_aligner, mode=self.mode)
        self.reuser.fit(x_train, y_train)

    def predict(self, user_data):
        return self.reuser.predict(user_data)