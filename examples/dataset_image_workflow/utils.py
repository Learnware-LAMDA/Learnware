import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from learnware.utils import choose_device


@torch.no_grad()
def evaluate(model, evaluate_set: Dataset, device=None, distribution=True):
    device = choose_device(0) if device is None else device
    if isinstance(model, nn.Module):
        model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total, correct, loss = 0, 0, torch.as_tensor(0.0, dtype=torch.float32, device=device)
    dataloader = DataLoader(evaluate_set, batch_size=1024, shuffle=True)
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        out = model(X) if isinstance(model, nn.Module) else model.predict(X)
        if not torch.is_tensor(out):
            out = torch.from_numpy(out).to(device)

        if distribution:
            loss += criterion(out, y)
            _, predicted = torch.max(out.data, 1)
        else:
            predicted = out

        total += y.size(0)
        correct += (predicted == y).sum().item()

    acc = correct / total * 100
    loss = loss / total

    if isinstance(model, nn.Module):
        model.train()

    return loss.item(), acc


def train_model(
    model: nn.Module,
    train_set: Dataset,
    valid_set: Dataset,
    save_path: str,
    epochs=35,
    batch_size=128,
    device=None,
    verbose=True,
):
    device = choose_device(0) if device is None else device

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    best_loss = 100000

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

            torch.save(model.state_dict(), save_path)
            if verbose:
                print("Epoch: {}, Valid Best Accuracy: {:.3f}% ({:.3f})".format(epoch + 1, valid_acc, valid_loss))
        if valid_acc > 99.0:
            if verbose:
                print("Early Stopping at 99% !")
            break

        if verbose and (epoch + 1) % 5 == 0:
            print(
                "Epoch: {}, Train Average Loss: {:.3f}, Accuracy {:.3f}%, Valid Average Loss: {:.3f}".format(
                    epoch + 1, np.mean(running_loss), train_acc, valid_loss
                )
            )
