import os
import joblib
import numpy as np
from learnware.model import BaseModel
import torch
from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
import torchtext.functional as F
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url


class Model(BaseModel):
    def __init__(self):
        super().__init__(input_shape=None, output_shape=(2,))
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_classes = 2
        input_dim = 768
        classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)
        self.model = XLMR_BASE_ENCODER.get_model(head=classifier_head).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(dir_path, "model.pth")))

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = sentence_preprocess(X)
        X = F.to_tensor(X, padding_value=1).to(self.device)
        return self.model(X)

    def finetune(self, X: np.ndarray, y: np.ndarray):
        pass


def sentence_preprocess(x_datapipe):
    padding_idx = 1
    bos_idx = 0
    eos_idx = 2
    max_seq_len = 256
    xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
    xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

    text_transform = T.Sequential(
        T.SentencePieceTokenizer(xlmr_spm_model_path),
        T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
        T.Truncate(max_seq_len - 2),
        T.AddToken(token=bos_idx, begin=True),
        T.AddToken(token=eos_idx, begin=False),
    )

    x_datapipe = [text_transform(x) for x in x_datapipe]
    # x_datapipe = x_datapipe.map(text_transform)
    return x_datapipe