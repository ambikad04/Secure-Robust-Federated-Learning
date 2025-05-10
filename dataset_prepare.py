# import sys

# sys.path.append("../")
import copy
import os
import pickle
import numpy as np
import random
import torch
import json
from path import Path
from argparse import ArgumentParser, Namespace
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, EMNIST, FashionMNIST
from torchvision import transforms
from flearn.data.dataset import (
    MNISTDataset,
    CIFARDataset,
    CIFAR100Dataset,
    EMNISTDataset,
    FashionMNISTDataset,
    DomainNet,
    DomainNetDataset,
    TinyImageNet,
    TinyImageNetDataset
)
from collections import Counter
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from flearn.utils.slicing import (
    noniid_slicing,
    random_slicing,
    noniid_dirichlet,
    fixed_noniid_dirichlet,
    quantity_based_label_imbalance,
    distribution_based_label_skew,
    quantity_skew,
)

from dotenv import load_dotenv

load_dotenv()

CURRENT_DIR = os.getenv("DATASET_DIR")
# /mnt/c/Users/S_G/Documents/GitHub/Dataset

DATASET = {
    "mnist": (MNIST, MNISTDataset),
    "cifar": (CIFAR10, CIFARDataset),
    "cifar10": (CIFAR10, CIFARDataset),
    "cifar100": (CIFAR100, CIFAR100Dataset),
    "emnist": (EMNIST, EMNISTDataset),
    "fashionmnist": (FashionMNIST, FashionMNISTDataset),
    "domainnet": (DomainNet, DomainNetDataset),
    "tinyimagenet": (TinyImageNet, TinyImageNetDataset)
}

DATASETS_TYPES = [
    "iid",
    "niid",
    "dniid",
    "synthetic",
    "mix",
    "qty_lbl_imb",  # non-iid with quantity based label imbalance
    "noiid_lbldir",  # non-iid with dirichilet based label imbalance
    "iid_diff_qty",  # quantity skew
]

SLICING = {
    "iid": random_slicing,
    "niid": noniid_slicing,
    "dniid": noniid_dirichlet,
    "qty_lbl_imb": quantity_based_label_imbalance,
    "noiid_lbldir": distribution_based_label_skew,
    "noiid_lbldir_b_5": distribution_based_label_skew,
    "iid_diff_qty": quantity_skew,
}


MEAN = {
    "mnist": (0.1307,),
    "cifar": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.4914, 0.4822, 0.4465),
    "emnist": (),
    "fashionmnist": (),
}

STD = {
    "mnist": (0.3015,),
    "cifar": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2023, 0.1994, 0.2010),
    "emnist": (),
    "fashionmnist": (),
}

SIZE = {
    "mnist": (28, 28),
    "cifar": (32, 32),
    "cifar100": (32, 32),
    "emnist": (28, 28),
    "fashionmnist": (28, 28),
    "domainnet": (64, 64),
    "tinyimagenet": (64, 64)
}


class MNISTDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        return sample, target


classes_per_client = None


def preprocess(args: Namespace) -> None:
    print(args)
    global classes_per_client
    classes_per_client = args.classes
    dataset_dir = f"{CURRENT_DIR}/{args.dataset}"
    pickles_dir = f"{CURRENT_DIR}/{args.dataset}/{args.type}/{args.classes}"
    if args.feature_noise != 0 and (args.beta != 0 and (args.type == "noiid_lbldir" or args.type == "iid_diff_qty")):
        pickles_dir = f"{CURRENT_DIR}/{args.dataset}/{args.type}_n_{str(args.feature_noise).split('.')[1]}_b_{str(args.beta).split('.')[1]}/{args.classes}"
    else:
        if args.feature_noise != 0:
            pickles_dir = f"{CURRENT_DIR}/{args.dataset}/{args.type}_n_{str(args.feature_noise).split('.')[1]}/{args.classes}"
        if args.beta != 0 and (args.type == "noiid_lbldir" or args.type == "iid_diff_qty"):
            pickles_dir = f"{CURRENT_DIR}/{args.dataset}/{args.type}_b_{str(args.beta).split('.')[1]}/{args.classes}"
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    num_train_clients = int(args.client_num_in_total)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(SIZE[args.dataset]),
            transforms.ToTensor(),
        ]
    )
    target_transform = None
    trainset_stats = {}
    testset_stats = {}

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.isdir(pickles_dir):
        # Create the directory structure
        os.makedirs(pickles_dir, exist_ok=True)
    ori_dataset, target_dataset = DATASET[args.dataset]
    if args.dataset == "emnist":
        trainset = ori_dataset(
            dataset_dir,
            split="balanced",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        testset = ori_dataset(
            dataset_dir, 
            split="balanced", 
            train=False, 
            transform=transforms.ToTensor()
        )
    elif args.dataset == "domainnet":
        trainset = ori_dataset(
            os.path.join(dataset_dir, 'raw_data'),
            domain = args.domain,
            train=True,
            download=True,
            transform=transform
        )
        testset = ori_dataset(
            os.path.join(dataset_dir, "raw_data"),
            domain=args.domain,
            train=False,
            transform=transform,
        )
    elif args.dataset == "tinyimagenet":
        trainset = ori_dataset(
            os.path.join(dataset_dir, 'raw_data'),
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        testset = ori_dataset(
            os.path.join(dataset_dir, "raw_data"),
            train=False,
            transform=transforms.ToTensor(),
        )
    else:
        trainset = ori_dataset(
            dataset_dir, 
            train=True, 
            download=True, 
            transform=transforms.ToTensor()
        )
        testset = ori_dataset(
            dataset_dir, 
            train=False, 
            transform=transforms.ToTensor()
        )

    noise_factor = args.feature_noise

    num_classes = args.total_classes
    distribution = args.type
    beta = args.beta
    all_trainsets, trainset_stats = randomly_alloc_classes(
        ori_dataset=trainset,
        target_dataset=target_dataset,
        num_clients=num_train_clients,
        num_classes=num_classes,
        num_classes_per_client=classes_per_client,
        transform=transform,
        target_transform=target_transform,
        distribution=distribution,
        noise_factor=noise_factor,
        beta= beta
    )

    client_id = args.start_idx
    for dataset in all_trainsets:
        with open(pickles_dir + f"/{client_id}.pkl", "wb") as f:
            pickle.dump(dataset, f)
        client_id += 1
    if args.dataset == "domainnet":
        if args.train_pkl:
            with open(pickles_dir + f"/train_{args.domain}.pkl", "wb") as f:
                pickle.dump(trainset, f)
        if args.test_pkl:
            with open(pickles_dir + f"/test_{args.domain}.pkl", "wb") as f:
                pickle.dump(testset, f)
    else:
        if args.train_pkl:
            with open(pickles_dir + f"/train.pkl", "wb") as f:
                pickle.dump(trainset, f)
        if args.test_pkl:
            with open(pickles_dir + f"/test.pkl", "wb") as f:
                pickle.dump(testset, f)
    with open(dataset_dir + f"/all_stats.json", "w") as f:
        json.dump({"train": trainset_stats, "test": testset_stats}, f)


def synthicet_data_preprocess(args: Namespace) -> None:
    print(args)
    global classes_per_client
    classes_per_client = args.classes
    dataset_dir = f"{CURRENT_DIR}/{args.dataset}"
    pickles_dir = f"{CURRENT_DIR}/{args.dataset}/{args.type}/{args.classes}"

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    num_train_clients = int(args.client_num_in_total * args.fraction)
    num_test_clients = args.client_num_in_total - num_train_clients

    # transform = transforms.Compose(
    #     [transforms.Normalize(MEAN[args.dataset], STD[args.dataset]),]
    # )
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(SIZE[args.dataset]),
            transforms.ToTensor(),
        ]
    )
    target_transform = None
    trainset_stats = {}
    testset_stats = {}

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.isdir(pickles_dir):
        # Create the directory structure
        os.makedirs(pickles_dir, exist_ok=True)
    ori_dataset, target_dataset = DATASET[args.dataset]
    if args.dataset == "emnist":
        trainset = ori_dataset(
            dataset_dir,
            split="balanced",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        testset = ori_dataset(
            dataset_dir, split="balanced", train=False, transform=transforms.ToTensor()
        )
    else:
        trainset = ori_dataset(
            dataset_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        testset = ori_dataset(dataset_dir, train=False, transform=transforms.ToTensor())

    num_classes = 10 if args.classes <= 0 else args.classes
    distribution = args.type
    all_trainsets, trainset_stats = create_sub_dataset(
        ori_dataset=trainset,
        target_dataset=target_dataset,
        num_clients=num_train_clients,
        num_classes=num_classes,
        transform=transform,
        target_transform=target_transform,
        distribution=distribution,
    )
    all_testsets, testset_stats = create_sub_dataset(
        ori_dataset=testset,
        target_dataset=target_dataset,
        num_clients=num_test_clients,
        num_classes=num_classes,
        transform=transform,
        target_transform=target_transform,
        distribution=distribution,
    )

    all_datasets = all_trainsets + all_testsets

    for client_id, dataset in enumerate(all_datasets):
        with open(pickles_dir + f"/{client_id}.pkl", "wb") as f:
            pickle.dump(dataset, f)
    with open(pickles_dir + f"/seperation.pkl", "wb") as f:
        pickle.dump(
            {
                "train": [i for i in range(num_train_clients)],
                "test": [i for i in range(num_train_clients, args.client_num_in_total)],
                "total": args.client_num_in_total,
            },
            f,
        )
    with open(dataset_dir + f"/all_stats.json", "w") as f:
        json.dump({"train": trainset_stats, "test": testset_stats}, f)


def randomly_alloc_classes(
    ori_dataset: Dataset,
    target_dataset: Dataset,
    num_clients: int,
    num_classes: int,
    num_classes_per_client: int,
    transform=None,
    target_transform=None,
    distribution: str = "niid",
    noise_factor: float = 0.0,
    beta: float=0.5,
) -> Tuple[List[Dataset], Dict[str, Dict[str, int]]]:
    print(f"num_clients={num_clients}, num_classes={num_classes}, distribution={distribution}")
    ori_dataset = add_feature_noise(
        ori_dataset, noise_factor
    )  # this converts the dataset to a list of tuples
    
    # Handle special case for noiid_lbldir_b_5
    if distribution == "noiid_lbldir_b_5":
        distribution = "noiid_lbldir"
        beta = 0.5  # Set beta to 0.5 for this specific case
    
    slicing = SLICING[distribution]
    if distribution == "niid":
        dict_users = slicing(ori_dataset, num_clients, num_clients * num_classes)
    if distribution == "dniid":
        dict_users = slicing(ori_dataset, num_clients, num_clients * num_classes)
    if distribution == "iid":
        dict_users = slicing(ori_dataset, num_clients)
    if distribution == "qty_lbl_imb":
        dict_users = slicing(
            ori_dataset, num_clients, num_classes, num_classes_per_client
        )
    if distribution == "noiid_lbldir":
        dict_users = slicing(ori_dataset, num_clients, num_classes, beta)
    if distribution == "iid_diff_qty":
        dict_users = slicing(ori_dataset, num_clients, beta)
    stats = {}
    for i, indices in dict_users.items():
        targets_numpy = np.array(ori_dataset.targets)
        # print(f'indices={indices}, type: {type(indices[0])}')
        stats[f"client {i}"] = {"x": 0, "y": {}}
        stats[f"client {i}"]["x"] = len(indices)
        stats[f"client {i}"]["y"] = Counter(targets_numpy[indices].tolist())
    datasets = []
    for indices in dict_users.values():
        datasets.append(
            target_dataset(
                [ori_dataset[i] for i in indices],
                transform=transform,
                target_transform=target_transform,
            )
        )
    return datasets, stats


def create_sub_dataset(
    ori_dataset: Dataset,
    target_dataset: Dataset,
    num_clients: int,
    num_classes: int,
    transform=None,
    target_transform=None,
    distribution: str = "niid",
) -> Tuple[List[Dataset], Dict[str, Dict[str, int]]]:
    total_sample_nums = len(ori_dataset.targets)
    sub_dataset_size = int(total_sample_nums / num_clients)
    created_dataset_dict = {}
    data_classes = list(range(len(ori_dataset.classes)))
    for i in range(num_clients):
        sub_indices = []
        ratios = None
        selected_classes = random.sample(data_classes, num_classes)
        if distribution != "iid":
            ratios = np.random.dirichlet(np.ones(len(selected_classes)))
        else:
            ratios = np.array(
                [1 / len(selected_classes) for _ in range(len(selected_classes))]
            )  # IID Data
        class_sample_size = (ratios * sub_dataset_size).astype(int)
        for data_class, sample_size in zip(selected_classes, class_sample_size):
            indices = [
                index
                for index, label in enumerate(ori_dataset.targets)
                if label == data_class
            ]
            sub_indices += random.sample(indices, min(sample_size, len(indices)))
        created_dataset_dict[i] = sub_indices
    stats = {}
    for i, indices in created_dataset_dict.items():
        targets_numpy = np.array(ori_dataset.targets)
        # print(f'indices={indices}, type: ')
        stats[f"client {i}"] = {"x": 0, "y": {}}
        stats[f"client {i}"]["x"] = len(indices)
        stats[f"client {i}"]["y"] = Counter(targets_numpy[indices].tolist())
    datasets = []
    for indices in created_dataset_dict.values():
        datasets.append(
            target_dataset(
                [ori_dataset[i] for i in indices],
                transform=transform,
                target_transform=target_transform,
            )
        )
    return datasets, stats


def add_feature_noise(ori_dataset, noise_factor):
    features = []
    labels = []
    for i in range(len(ori_dataset)):
        noise_pixel = ori_dataset[i][0] + noise_factor * torch.randn(
            ori_dataset[i][0].shape
        )
        noise_pixel = torch.clamp(noise_pixel, 0, 1)
        label = ori_dataset[i][1]
        features.append(noise_pixel)
        labels.append(label)
    features = torch.stack(features)
    labels = torch.tensor(labels)
    return MNISTDataset(features, labels)


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--type', type=str, default='iid')
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--client_num_in_total', type=int, default=10)
    parser.add_argument('--total_classes', type=int, default=10)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--feature_noise', type=float, default=0.0)
    parser.add_argument('--train_pkl', action='store_true')
    parser.add_argument('--test_pkl', action='store_true')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--domain', type=str, default='clipart')
    
    args = parser.parse_args()
    preprocess(args)

if __name__ == '__main__':
    main()
