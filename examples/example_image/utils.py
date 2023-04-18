import os
import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim

from model import ConvModel


class ImageDataLoader:
    def __init__(self, data_root, train: bool = True):
        self.data_root = data_root
        self.train = train

    def get_idx_data(self, idx=0):
        if self.train:
            X_path = os.path.join(self.data_root, "uploader", "uploader_%d_X.npy" % (idx))
            y_path = os.path.join(self.data_root, "uploader", "uploader_%d_y.npy" % (idx))
            if not (os.path.exists(X_path) and os.path.exists(y_path)):
                raise Exception("Index Error")
            X = np.load(X_path)
            y = np.load(y_path)
        else:
            X_path = os.path.join(self.data_root, "user", "user_%d_X.npy" % (idx))
            y_path = os.path.join(self.data_root, "user", "user_%d_y.npy" % (idx))
            if not (os.path.exists(X_path) and os.path.exists(y_path)):
                raise Exception("Index Error")
            X = np.load(X_path)
            y = np.load(y_path)
        return X, y


def generate_uploader(data_x, data_y, n_uploaders=50, data_save_root=None):
    if data_save_root is None:
        return
    os.makedirs(data_save_root, exist_ok=True)
    for i in range(n_uploaders):
        random_class_num = random.randint(6, 10)
        cls_indx = list(range(10))
        random.shuffle(cls_indx)
        selected_cls_indx = cls_indx[:random_class_num]
        rest_cls_indx = cls_indx[random_class_num:]
        selected_data_indx = []
        for cls in selected_cls_indx:
            data_indx = list(torch.where(data_y == cls)[0])
            # print(type(data_indx))
            random.shuffle(data_indx)
            data_num = random.randint(800, 2000)
            selected_indx = data_indx[:data_num]
            selected_data_indx = selected_data_indx + selected_indx
        for cls in rest_cls_indx:
            flag = random.randint(0, 1)
            if flag == 0:
                continue
            data_indx = list(torch.where(data_y == cls)[0])
            random.shuffle(data_indx)
            data_num = random.randint(20, 80)
            selected_indx = data_indx[:data_num]
            selected_data_indx = selected_data_indx + selected_indx
        selected_X = data_x[selected_data_indx].numpy()
        selected_y = data_y[selected_data_indx].numpy()
        print(selected_X.dtype, selected_y.dtype)
        print(selected_X.shape, selected_y.shape)
        X_save_dir = os.path.join(data_save_root, "uploader_%d_X.npy" % (i))
        y_save_dir = os.path.join(data_save_root, "uploader_%d_y.npy" % (i))
        np.save(X_save_dir, selected_X)
        np.save(y_save_dir, selected_y)
        print("Saving to %s" % (X_save_dir))


def generate_user(data_x, data_y, n_users=50, data_save_root=None):
    if data_save_root is None:
        return
    os.makedirs(data_save_root, exist_ok=True)
    for i in range(n_users):
        random_class_num = random.randint(3, 6)
        cls_indx = list(range(10))
        random.shuffle(cls_indx)
        selected_cls_indx = cls_indx[:random_class_num]
        selected_data_indx = []
        for cls in selected_cls_indx:
            data_indx = list(torch.where(data_y == cls)[0])
            # print(type(data_indx))
            random.shuffle(data_indx)
            data_num = random.randint(150, 350)
            selected_indx = data_indx[:data_num]
            selected_data_indx = selected_data_indx + selected_indx
        # print('Total Index:', len(selected_data_indx))
        selected_X = data_x[selected_data_indx].numpy()
        selected_y = data_y[selected_data_indx].numpy()
        print(selected_X.shape, selected_y.shape)
        X_save_dir = os.path.join(data_save_root, "user_%d_X.npy" % (i))
        y_save_dir = os.path.join(data_save_root, "user_%d_y.npy" % (i))
        np.save(X_save_dir, selected_X)
        np.save(y_save_dir, selected_y)
        print("Saving to %s" % (X_save_dir))


# Train Uploaders' models
def train(X, y, out_classes, epochs=35, batch_size=128):
    print(X.shape, y.shape)
    input_feature = X.shape[1]
    data_size = X.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvModel(channel=input_feature, n_random_features=out_classes).to(device)
    model.train()

    # Adam optimizer with learning rate 1e-3
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # SGD optimizer with learning rate 1e-2
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    # mean-squared error loss
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = []
        indx = list(range(data_size))
        random.shuffle(indx)
        curr_X = X[indx]
        curr_y = y[indx]
        for i in range(math.floor(data_size / batch_size)):
            inputs, annos = curr_X[i * batch_size : (i + 1) * batch_size], curr_y[i * batch_size : (i + 1) * batch_size]
            inputs = torch.from_numpy(inputs).to(device)
            annos = torch.from_numpy(annos).to(device)
            # print(inputs.dtype, annos.dtype)
            out = model(inputs)
            optimizer.zero_grad()
            loss = criterion(out, annos)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        # print('Epoch: %d, Average Loss: %.3f'%(epoch+1, np.mean(running_loss)))

    # Train Accuracy
    acc = test(X, y, model)
    model.train()
    return model


def test(test_X, test_y, model, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total, correct = 0, 0
    data_size = test_X.shape[0]
    for i in range(math.ceil(data_size / batch_size)):
        inputs, annos = test_X[i * batch_size : (i + 1) * batch_size], test_y[i * batch_size : (i + 1) * batch_size]
        inputs = torch.Tensor(inputs).to(device)
        annos = torch.Tensor(annos).to(device)
        out = model(inputs)
        _, predicted = torch.max(out.data, 1)
        total += annos.size(0)
        correct += (predicted == annos).sum().item()
    acc = correct / total * 100
    print("Accuracy: %.2f" % (acc))
    return acc


def eval_prediction(pred_y, target_y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, predicted = torch.max(pred_y.data, 1)
    annos = torch.from_numpy(target_y).to(device)
    total = annos.size(0)
    correct = (predicted == annos).sum().item()
    criterion = nn.CrossEntropyLoss()
    return correct / total
