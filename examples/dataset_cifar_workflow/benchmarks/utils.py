import json
import os
import zipfile
from collections import defaultdict
from shutil import rmtree
from tabulate import tabulate

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset

from learnware.client import LearnwareClient
from learnware.learnware import Learnware
from learnware.specification import generate_rkme_image_spec, RKMEImageSpecification
from .dataset import uploader_data, user_data
from .models.conv import ConvModel
from learnware.market import LearnwareMarket
from learnware.utils import choose_device

@torch.no_grad()
def evaluate(model, evaluate_set: Dataset, device=None):
    device = choose_device(0) if device is None else device

    if isinstance(model, nn.Module):
        model.eval()
        mapping = lambda m, x: m(x)
    else: # For predict interface
        mapping = lambda m, x: m.predict(x)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total, correct, loss = 0, 0, 0.0
    dataloader = DataLoader(evaluate_set, batch_size=512, shuffle=True)
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        out = mapping(model, X)
        if not torch.is_tensor(out):
            out = torch.from_numpy(out).to(device)
        loss += criterion(out, y)

        _, predicted = torch.max(out.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    acc = correct / total * 100
    loss = loss / total

    if isinstance(model, nn.Module):
        model.train()

    return loss.item(), acc


def build_learnware(name: str, market: LearnwareMarket, order, model_name="conv",
                    out_classes=10, epochs=35, batch_size=128, device=None):
    device = choose_device(0) if device is None else device

    if name == "cifar10":
        train_set, valid_set, spec_set, order = uploader_data(order=order)
    else:
        raise Exception("Not support", name)

    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cache', 'learnware'))
    if os.path.exists(cache_dir):
        rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    channel = train_set[0][0].shape[0]
    image_size = train_set[0][0].shape[1], train_set[0][0].shape[2]

    model = ConvModel(channel=channel, im_size=image_size,
                      n_random_features=out_classes).to(device)

    model.train()

    # SGD optimizer with learning rate 1e-2
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    # Scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    # mean-squared error loss
    criterion = nn.CrossEntropyLoss()
    # Prepare DataLoader
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # valid loss
    best_loss = 100000 # initially
    # Optimizing...
    for epoch in range(epochs):
        running_loss = []
        model.train()
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device=device), y.to(device=device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        valid_loss, valid_acc = evaluate(model, valid_set, device=device)
        train_loss, train_acc = evaluate(model, train_set, device=device)
        if valid_loss < best_loss:
            best_loss = valid_loss

            torch.save(model.state_dict(), os.path.join(cache_dir, "model.pth"))
            print("Epoch: {}, Valid Best Accuracy: {:.3f}% ({:.3f})".format(epoch+1, valid_acc, valid_loss))
        if valid_acc > 99.0:
            print("Early Stopping at 99% !")
            break

        if (epoch + 1) % 5 == 0:
            print('Epoch: {}, Train Average Loss: {:.3f}, Accuracy {:.3f}%, Valid Average Loss: {:.3f}'.format(
                epoch+1, np.mean(running_loss), train_acc, valid_loss))

        # scheduler.step()

    # build specification
    loader = DataLoader(spec_set, batch_size=3000, shuffle=True)
    sampled_X, _ = next(iter(loader))
    spec = generate_rkme_image_spec(sampled_X, whitening=False)

    # add to market
    model_dir = os.path.abspath(os.path.join(__file__, "..", "models"))
    spec.save(os.path.join(cache_dir, "spec.json"))

    zip_file = os.path.join(cache_dir, "learnware.zip")
    # zip -q -r -j zip_file dir_path
    with zipfile.ZipFile(zip_file, "w") as zip_obj:
        for foldername, subfolders, filenames in os.walk(os.path.join(model_dir, model_name)):
            for filename in filenames:
                if filename.endswith(".pyc"):
                    continue
                file_path = os.path.join(foldername, filename)
                zip_info = zipfile.ZipInfo(filename)
                zip_info.compress_type = zipfile.ZIP_STORED
                with open(file_path, "rb") as file:
                    zip_obj.writestr(zip_info, file.read())

        for filename, file_path in zip(["spec.json", "model.pth", "learnware.yaml"],
                                      [os.path.join(cache_dir, "spec.json"),
                                       os.path.join(cache_dir, "model.pth"),
                                       os.path.join(model_dir, "learnware.yaml")]):
            zip_info = zipfile.ZipInfo(filename)
            zip_info.compress_type = zipfile.ZIP_STORED
            with open(file_path, "rb") as file:
                zip_obj.writestr(zip_info, file.read())

    print(", ".join([str(o) for o in order]))
    market.add_learnware(zip_file, semantic_spec=LearnwareClient.create_semantic_specification(
        self=None,
        name="learnware",
        description=", ".join([str(o) for o in order]),
        data_type="Image",
        task_type="Classification",
        library_type="PyTorch",
        scenarios=["Computer"],
        output_description={"Dimension": out_classes, "Description": {str(i): "i" for i in range(out_classes)}})
    )

    return model


def build_specification(name: str, cache_id, order, sampled_size=3000):
    cache_dir = os.path.abspath(os.path.join(
        os.path.dirname( __file__ ), '..', 'cache', 'spec'))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "spec_{}.json".format(cache_id))

    if os.path.exists(cache_path):
        spec = RKMEImageSpecification()
        spec.load(cache_path)

        test_dataset, spec_dataset, _, _ = user_data(indices=torch.asarray(spec.msg))
    else:
        test_dataset, spec_dataset, indices, _ = user_data(order=order)
        loader = DataLoader(spec_dataset, batch_size=sampled_size, shuffle=True)
        sampled_X, _ = next(iter(loader))
        spec = generate_rkme_image_spec(sampled_X, whitening=False)

        spec.msg = indices.tolist()
        spec.save(cache_path)

    return spec, test_dataset


class Recorder:

    def __init__(self):
        self.data = defaultdict(list)

    def record(self, name, accuracy, loss):
        self.data[name].append((accuracy, loss))

    def latest(self):
        table = []

        for name, values in self.data.items():
            value = values[-1]
            table.append([name, "{:.3f}%".format(value[0]), "{:.3f}".format(value[1])])

        return str(tabulate(table, headers=["Case", "Accuracy", "Loss"], tablefmt='orgtbl'))

    def accumulated(self):
        table = []

        for name, values in self.data.items():
            value_mean = [np.mean(v) for v in zip(*values)]
            value_std = [np.std(v) for v in zip(*values)]
            table.append([name,
                          "{:.3f}% ± {:.3f}%".format(value_mean[0], value_std[0]),
                          "{:.3f} ± {:.3f}" .format(value_mean[1], value_std[1])])

        return str(tabulate(table, headers=["Case", "Accuracy", "Loss"], tablefmt='orgtbl'))