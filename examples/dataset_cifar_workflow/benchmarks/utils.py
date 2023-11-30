import os
import zipfile
from shutil import rmtree

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
    elif isinstance(model, Learnware):
        mapping = lambda m, x: m.predict(x)
    else:
        raise Exception("not support model type", model)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total, correct, loss = 0, 0, 0.0
    dataloader = DataLoader(evaluate_set, batch_size=512, shuffle=True)
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        out = mapping(model, X)
        loss += criterion(out, y)

        _, predicted = torch.max(out.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    acc = correct / total * 100
    loss = loss / total

    if isinstance(model, nn.Module):
        model.train()

    return loss, acc


def build_learnware(name: str, market: LearnwareMarket, model_name="conv",
                    out_classes=10, epochs=200, batch_size=2048, device=None):
    device = choose_device(0) if device is None else device

    if name == "cifar10":
        train_set, valid_set, spec_set = uploader_data()
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
    # if device.type == 'cuda':
    #     model = nn.DataParallel(model)
    #     model.benchmark = True

    model.train()

    # SGD optimizer with learning rate 1e-2
    optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
    # Scheduler TODO: Use this
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
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device=device), y.to(device=device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        valid_loss, valid_acc = evaluate(model, train_set, device=device)
        if valid_loss < best_loss:
            best_loss = valid_loss
            if isinstance(model, nn.DataParallel):
                model_to_save = model.module
            else:
                model_to_save = model
            torch.save(model_to_save.state_dict(), os.path.join(cache_dir, "model.pth"))
            print("Epoch: {}, Valid Best Accuracy: {:.3f}% ({:.3f})".format(epoch+1, valid_acc, valid_loss))

        if (epoch + 1) % 5 == 0:
            print('Epoch: {}, Train Average Loss: {:.3f}, Valid Average Loss: {:.3f}'.format(
                epoch+1, np.mean(running_loss), valid_loss))

        # scheduler.step()

    # build specification
    loader = DataLoader(spec_set, batch_size=3000, shuffle=True)
    sampled_X, _ = next(iter(loader))
    spec = generate_rkme_image_spec(sampled_X)

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

    market.add_learnware(zip_file, semantic_spec=LearnwareClient.create_semantic_specification(
        self=None,
        name="learnware",
        description="For Cifar Dataset Workflow",
        data_type="Image",
        task_type="Classification",
        library_type="PyTorch",
        scenarios=["Computer"],
        output_description={"Dimension": out_classes, "Description": {str(i): "i" for i in range(out_classes)}})
    )

    return model


def build_specification(name: str, cache_id, sampled_size=3000):
    cache_path = os.path.abspath(os.path.join(
        os.path.dirname( __file__ ), '..', 'cache', "{}.json".format(cache_id)))

    if name == "cifar10":
        dataset = user_data()
    else:
        raise Exception("Not support", name)

    if os.path.exists(cache_path):
        spec = RKMEImageSpecification()
        spec.load(cache_path)
        return spec, dataset

    loader = DataLoader(dataset, batch_size=sampled_size, shuffle=True)
    sampled_X, _ = next(iter(loader))
    spec = generate_rkme_image_spec(sampled_X, steps=1)

    spec.save(cache_path)
    return spec, dataset