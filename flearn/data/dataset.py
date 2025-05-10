import os
from typing import Callable, Optional
import numpy as np
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive

CLASSES = {"mnist":10, "cifar":10, "cifar100":100, "tinyimagenet":200, "emnist": 47, "fashionmnist":10}

class MNISTDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: tup[0]
                        if isinstance(tup[0], torch.Tensor)
                        else torch.tensor(tup[0]),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: tup[1]
                        if isinstance(tup[1], torch.Tensor)
                        else torch.tensor(tup[1]),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return data, targets

    def __len__(self):
        return len(self.targets)

class CIFARDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: tup[0]
                        if isinstance(tup[0], torch.Tensor)
                        else torch.tensor(tup[0]),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: tup[1]
                        if isinstance(tup[1], torch.Tensor)
                        else torch.tensor(tup[1]),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return img, targets

    def __len__(self):
        return len(self.targets)

class CIFAR100Dataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: tup[0]
                        if isinstance(tup[0], torch.Tensor)
                        else torch.tensor(tup[0]),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: tup[1]
                        if isinstance(tup[1], torch.Tensor)
                        else torch.tensor(tup[1]),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return img, targets

    def __len__(self):
        return len(self.targets)

class EMNISTDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: tup[0]
                        if isinstance(tup[0], torch.Tensor)
                        else torch.tensor(tup[0]),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: tup[1]
                        if isinstance(tup[1], torch.Tensor)
                        else torch.tensor(tup[1]),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return data, targets

    def __len__(self):
        return len(self.targets)

class FashionMNISTDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: tup[0]
                        if isinstance(tup[0], torch.Tensor)
                        else torch.tensor(tup[0]),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: tup[1]
                        if isinstance(tup[1], torch.Tensor)
                        else torch.tensor(tup[1]),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return data, targets

    def __len__(self):
        return len(self.targets)    

class DomainNet(Dataset):
    dndl_links = {
        "clipart": {
            "link": "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
            "train": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt",
            "test": "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_test.txt",
        },
        "infograph": {
            "link": "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
            "train": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt",
            "test": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt",
        },
        "painting": {
            "link": "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
            "train": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt",
            "test": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt",
        },
        "quickdraw": {
            "link": "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
            "train": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt",
            "test": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt",
        },
        "real": {
            "link": "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
            "train": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt",
            "test": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt",
        },
        "sketch": {
            "link": "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
            "train": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt",
            "test": "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt",
        },
    }
    def __init__(
        self,
        root: str,
        domain = "all",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        train: bool = True,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        if domain == "all":
            self.domain = self.dndl_links.keys()
        else:
            self.domain = [domain]
        self.train = train
        if download:
            self.download()
        self.data = []
        self.targets = []
        self._load_data()

    def download(self):
        # check whether the dataset is downloaded
        if os.path.exists(self.root):
            print("Dataset already downloaded.")
        else:
            print("Dataset not found. Downloading...")
        os.makedirs(self.root, exist_ok=True)
        for domain in self.domain:
            if domain not in self.dndl_links:
                raise ValueError(f"Domain {domain} not found in DomainNet")
            download_and_extract_archive(self.dndl_links[domain]['link'], self.root)

    def _load_data(self):
        for domain in self.domain:
            print(f"Loading {domain} data. Train: {self.train}")
            if self.train:
                # download the train file
                response = requests.get(self.dndl_links[domain]["train"])
                response.raise_for_status() 
                for line in response.text.splitlines():
                    img_path, label = line.split()
                    full_path = os.path.join(self.root, img_path)
                    img = Image.open(full_path)
                    self.data.append(np.array(img))
                    self.targets.append(int(label))
            else:
                response = requests.get(self.dndl_links[domain]["test"])
                response.raise_for_status()
                for line in response.text.splitlines():
                    img_path, label = line.split()
                    full_path = os.path.join(self.root, img_path)
                    img = Image.open(full_path)
                    self.data.append(np.array(img))
                    self.targets.append(int(label))

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return data, targets

    def __len__(self):
        return len(self.targets)


class DomainNetDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[0]
                            if isinstance(tup[0], torch.Tensor)
                            else torch.tensor(tup[0])
                        ),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[1]
                            if isinstance(tup[1], torch.Tensor)
                            else torch.tensor(tup[1])
                        ),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return img, targets


class TinyImageNet(Dataset):
    dndl_links = {
        "data": {
            "link": "https://github.com/tjmoon0104/pytorch-tiny-imagenet/releases/download/tiny-imagenet-dataset/processed-tiny-imagenet-200.zip",
            "path": "tiny-imagenet-200"
        }
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        train: bool = True,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.train = train

        # Download dataset if needed
        if download:
            self.download()

        # Get the appropriate path
        if self.train:
            self.data_dir = os.path.join(self.root, self.dndl_links["data"]["path"], "train")
        else:
            self.data_dir = os.path.join(self.root, self.dndl_links["data"]["path"], "val")

        # Map class names to integer labels (0 to 199)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(self.data_dir)))}

        # Load data
        self.data = []
        self.targets = []
        self._load_data()

    def download(self):
        if os.path.exists(os.path.join(self.root, self.dndl_links['data']['path'])):
            print("Dataset already downloaded.")
        else:
            print("Dataset not found. Downloading...")
            os.makedirs(self.root, exist_ok=True)
            download_and_extract_archive(self.dndl_links['data']['link'], self.root)

    def _load_data(self):
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.data_dir, class_name, "images")
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.data.append(img_path)
                self.targets.append(class_idx)

    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)

class TinyImageNetDataset(Dataset):
    def __init__(
        self,
        subset=None,
        data=None,
        targets=None,
        transform=None,
        target_transform=None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        if (data is not None) and (targets is not None):
            self.data = data.unsqueeze(1)
            self.targets = targets
        elif subset is not None:
            self.data = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[0]
                            if isinstance(tup[0], torch.Tensor)
                            else torch.tensor(tup[0])
                        ),
                        subset,
                    )
                )
            )
            self.targets = torch.stack(
                list(
                    map(
                        lambda tup: (
                            tup[1]
                            if isinstance(tup[1], torch.Tensor)
                            else torch.tensor(tup[1])
                        ),
                        subset,
                    )
                )
            )
        else:
            raise ValueError(
                "Data Format: subset: Tuple(data: Tensor / Image / np.ndarray, targets: Tensor) OR data: List[Tensor]  targets: List[Tensor]"
            )

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return img, targets
    
    def __len__(self):
        return len(self.targets)

    def get_trainloader(self, batch_size):
        return DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Add multiple worker processes
            pin_memory=True  # Pin memory for faster GPU transfer
        )

    def get_testloader(self, batch_size):
        return DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,  # Add multiple worker processes
            pin_memory=True  # Pin memory for faster GPU transfer
        )