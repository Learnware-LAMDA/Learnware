from torchvision import datasets


def cifar10(split="uploader"):
    assert(split in {"uploader", "user"})

    if split == "uploader":
        dataset = datasets.CIFAR10(root="cache", download=True, train=True)
    else:
        dataset = datasets.CIFAR10(root="cache", download=True, train=False)