import os
import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim

import pickle
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url
from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
import torchtext.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader



class TextDataLoader:
    def __init__(self, data_root, train: bool = True):
        self.data_root = data_root
        self.train = train

    def get_idx_data(self, idx=0):
        if self.train:
            X_path = os.path.join(self.data_root, "uploader", "uploader_%d_X.pkl" % (idx))
            y_path = os.path.join(self.data_root, "uploader", "uploader_%d_y.pkl" % (idx))
            if not (os.path.exists(X_path) and os.path.exists(y_path)):
                raise Exception("Index Error")
            with open(X_path, 'rb') as f:
                X = pickle.load(f)
            with open(y_path, 'rb') as f:
                y = pickle.load(f)
        else:
            X_path = os.path.join(self.data_root, "user", "user_%d_X.pkl" % (idx))
            y_path = os.path.join(self.data_root, "user", "user_%d_y.pkl" % (idx))
            if not (os.path.exists(X_path) and os.path.exists(y_path)):
                raise Exception("Index Error")
            with open(X_path, 'rb') as f:
                X = pickle.load(f)
            with open(y_path, 'rb') as f:
                y = pickle.load(f)
        return X, y


def generate_uploader(data_x, data_y, n_uploaders=50, data_save_root=None):
    if data_save_root is None:
        return
    os.makedirs(data_save_root, exist_ok=True)
    n = len(data_x)
    for i in range(n_uploaders):
        selected_X = data_x[i * (n // n_uploaders): (i + 1) * (n // n_uploaders)]
        selected_y = data_y[i * (n // n_uploaders): (i + 1) * (n // n_uploaders)]
        X_save_dir = os.path.join(data_save_root, "uploader_%d_X.pkl" % (i))
        y_save_dir = os.path.join(data_save_root, "uploader_%d_y.pkl" % (i))
        with open(X_save_dir, 'wb') as f:
            pickle.dump(selected_X, f)
        with open(y_save_dir, 'wb') as f:
            pickle.dump(selected_y, f)
        print("Saving to %s" % (X_save_dir))


def generate_user(data_x, data_y, n_users=50, data_save_root=None):
    if data_save_root is None:
        return
    os.makedirs(data_save_root, exist_ok=True)
    n = len(data_x)
    for i in range(n_users):
        selected_X = data_x[i * (n // n_users): (i + 1) * (n // n_users)]
        selected_y = data_y[i * (n // n_users): (i + 1) * (n // n_users)]
        X_save_dir = os.path.join(data_save_root, "user_%d_X.pkl" % (i))
        y_save_dir = os.path.join(data_save_root, "user_%d_y.pkl" % (i))
        with open(X_save_dir, 'wb') as f:
            pickle.dump(selected_X, f)
        with open(y_save_dir, 'wb') as f:
            pickle.dump(selected_y, f)
        print("Saving to %s" % (X_save_dir))


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


def train_step(model, criteria, optim, input, target):
    output = model(input)
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()


def eval_step(model, criteria, input, target):
    output = model(input)
    loss = criteria(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()


def evaluate(model, criteria, dev_dataloader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for batch in dev_dataloader:
            input = F.to_tensor(batch["token_ids"], padding_value=1).to(DEVICE)
            target = torch.tensor(batch["target"]).to(DEVICE)
            loss, predictions = eval_step(model, criteria, input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train Uploaders' models
def train(X, y, out_classes, epochs=35, batch_size=128):
    # print(X.shape, y.shape)
    from torchdata.datapipes.iter import IterableWrapper
    X = sentence_preprocess(X)
    data_size = len(X)
    train_datapipe = list(zip(X, y))
    train_datapipe = IterableWrapper(train_datapipe)
    train_datapipe = train_datapipe.batch(batch_size)
    train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
    train_dataloader = DataLoader(train_datapipe, batch_size=None)

    num_classes = 2
    input_dim = 768
    classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)
    model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
    learning_rate = 1e-5
    optim = AdamW(model.parameters(), lr=learning_rate)
    criteria = nn.CrossEntropyLoss()

    model.to(DEVICE)

    num_epochs = 10

    for e in range(num_epochs):
        for batch in train_dataloader:
            input = F.to_tensor(batch["token_ids"], padding_value=1).to(DEVICE)
            target = torch.tensor(batch["target"]).to(DEVICE)
            train_step(model, criteria, optim, input, target)

        loss, accuracy = evaluate(model, criteria, train_dataloader)
        print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, accuracy))
    return model


def eval_prediction(pred_y, target_y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(pred_y, np.ndarray):
        pred_y = pred_y.detach().cpu().numpy()
    if len(pred_y.shape) == 1:
        predicted = np.array(pred_y)
    else:
        predicted = np.argmax(pred_y, 1)
    annos = np.array(target_y)
    # print(predicted, annos)
    # annos = target_y
    total = predicted.shape[0]
    correct = (predicted == annos).sum().item()
    criterion = nn.CrossEntropyLoss()
    return correct / total
