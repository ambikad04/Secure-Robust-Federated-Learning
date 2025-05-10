# @sujit880
"""functions associated with data and dataset slicing"""

import random
import warnings
import numpy as np
from tqdm import trange


def noniid_slicing(dataset, num_clients, num_shards):
    """Slice a dataset for non-IID.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset to slice.
        num_clients (int):  Number of client.
        num_shards (int): Number of shards.
    
    Notes:
        The size of a shard equals to ``int(len(dataset)/num_shards)``.
        Each client will get ``int(num_shards/num_clients)`` shards.

    Returns：
        dict: ``{ 0: indices of dataset, 1: indices of dataset, ..., k: indices of dataset }``
    """
    total_sample_nums = len(dataset)
    size_of_shards = int(total_sample_nums / num_shards)
    if total_sample_nums % num_shards != 0:
        warnings.warn(
            "warning: the length of dataset isn't divided exactly by num_shard.some samples will be dropped."
        )
    # the number of shards that each one of clients can get
    shard_pc = int(num_shards / num_clients)
    if num_shards % num_clients != 0:
        warnings.warn(
            "warning: num_shard isn't divided exactly by num_clients. some samples will be dropped."
        )

    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}

    labels = np.array(dataset.targets)
    idxs = np.arange(total_sample_nums)

    # sort sample indices according to labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]  # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]

    # assign
    idx_shard = [i for i in range(num_shards)]
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, shard_pc, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i],
                 idxs[rand * size_of_shards:(rand + 1) * size_of_shards]),
                axis=0)

    return dict_users


def random_slicing(dataset, num_clients):
    """Slice a dataset randomly and equally for IID.

    Args：
        dataset (torch.utils.data.Dataset): a dataset for slicing.
        num_clients (int):  the number of client.

    Returns：
        dict: ``{ 0: indices of dataset, 1: indices of dataset, ..., k: indices of dataset }``
    """
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in trange(num_clients):
        dict_users[i] = list(
            np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users


def noniid_dirichlet(dataset, num_clients, num_shards):
    """Slice a dataset for non-IID.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset to slice.
        num_clients (int):  Number of clients.
        num_shards (int): Number of shards.
    
    Notes:
        The size of a shard equals to ``int(len(dataset)/num_shards)``.
        Each client will get ``int(num_shards/num_clients)`` shards.

    Returns:
        dict: ``{ 0: indices of dataset, 1: indices of dataset, ..., k: indices of dataset }``
    """
    total_sample_nums = len(dataset)
    size_of_shards = int(total_sample_nums / num_shards)
    if total_sample_nums % num_shards != 0:
        warnings.warn(
            "warning: the length of dataset isn't divided exactly by num_shard. Some samples will be dropped."
        )
    # the number of shards that each one of clients can get
    shard_pc = int(num_shards / num_clients)
    if num_shards % num_clients != 0:
        warnings.warn(
            "warning: num_shard isn't divided exactly by num_clients. Some samples will be dropped."
        )

    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}

    labels = np.array(dataset.targets)
    idxs = np.arange(total_sample_nums)

    # sort sample indices according to labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]  # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]

    # Assign samples to clients using Dirichlet distribution parameters
    dirichlet_parameters = np.random.uniform(1, 10, size=(num_clients, len(np.unique(labels))))
    for i in trange(num_clients):
        # Generate sample counts for each label using Dirichlet distribution
        sample_counts = np.random.dirichlet(dirichlet_parameters[i], size=1).flatten() * size_of_shards * shard_pc
        # Convert to integer counts
        sample_counts = np.floor(sample_counts).astype(int)
        
        # Assign samples to clients
        start = 0
        for label, count in enumerate(sample_counts):
            # Get indices of samples with the given label
            label_idxs = idxs[labels == label]
            # Randomly sample 'count' number of samples
            sampled_idxs = np.random.choice(label_idxs, count, replace=False)
            dict_users[i] = np.concatenate((dict_users[i], sampled_idxs), axis=0)

    return dict_users

def fixed_noniid_dirichlet(dataset, num_clients, num_shards, num_classes_per_client):
    """Slice a dataset for non-IID using Dirichlet distribution.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to slice.
        num_clients (int): Number of clients.
        num_shards (int): Number of shards.
        num_classes_per_client (int): Number of classes per client (desired, not guaranteed).

    Returns:
        dict: {0: indices of dataset, 1: indices of dataset, ..., k: indices of dataset}
    """
    # num_clients = 20
    total_sample_nums = len(dataset)
    size_of_shards = int(total_sample_nums / num_shards)

    # Handle potential remainder in dataset length
    remainder = total_sample_nums % num_shards
    shard_sizes = [size_of_shards] * num_shards
    for i in range(remainder):
        shard_sizes[i] += 1  # Assign remaining samples to first 'remainder' shards

    # Initialize client data structures
    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}
    labels = np.array(dataset.targets)
    idxs = np.arange(total_sample_nums)

    # Sort samples by labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    print(f'idxs: {idxs[0]}, {idxs_labels[1][1]}, idxs_labels: {len(idxs_labels[0])}')
    # Dirichlet distribution parameters (one per client)
    dirichlet_parameters = np.random.uniform(1, 5, size=(num_clients, num_classes_per_client))
    assigned_classes_params = np.random.randint(1, 9, size=(num_clients, len(np.unique(labels))))
    # Loop through clients
    for i in trange(num_clients):
        assigned_count = 0  # Track total samples assigned
        shard_pc = int(num_shards / num_clients)
        # Sample class from Dirichlet distribution (might not be unique)
        sampled_class = np.argmax(np.random.dirichlet(dirichlet_parameters[i]))
        sample_counts = np.random.dirichlet(dirichlet_parameters[i], size=1).flatten() * size_of_shards * shard_pc
        assign_sample_counts = np.random.dirichlet(assigned_classes_params[i], size=1).flatten() * size_of_shards * shard_pc
        # Convert to integer counts
        sample_counts = np.floor(sample_counts).astype(int)
        assigned_classes = np.argsort(assign_sample_counts.astype(int))[-2:]
        max_values = assign_sample_counts[assigned_classes]
        print(f"sampled_class: {sampled_class}, \n sample_counts: {sample_counts}, \nmax_indices: {assigned_classes} ,max_values: {max_values}")
        # Assign samples to clients
        start = 0
        for label, count in zip(assigned_classes ,sample_counts):
            print("lc",label, count)
            print(type(label), type(count))
            # Get indices of samples with the given label
            label_idxs = idxs[labels == label]
            # Randomly sample 'count' number of samples
            sampled_idxs = np.random.choice(label_idxs, count, replace=False)
            print(type(sampled_idxs[0]))
            dict_users[i] = np.concatenate((dict_users[i], sampled_idxs), axis=0)

    return dict_users

def quantity_based_label_imbalance(dataset, num_clients, num_classes, num_classes_per_client):
    labels = np.array(dataset.targets)
    
    # distribute labels among n_parties
    times = [0 for _ in range(num_classes)]
    contain = []
    for i in trange(num_clients):
        current = [i%num_classes]
        times[i%num_classes] += 1
        j = 1
        while j < num_classes_per_client:
            ind = random.randint(0, num_classes-1)
            if ind not in current:
                j += 1
                current.append(ind)
                times[ind] += 1
        contain.append(current)
    
    # assign samples of selected labels to clients
    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}
    for i in trange(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        split = np.array_split(idxs, times[i])
        ids = 0
        for j in range(num_clients):
            if i in contain[j]:
                dict_users[j] = np.append(dict_users[j], split[ids])
                ids += 1
    return dict_users

def distribution_based_label_skew(dataset, num_clients, num_classes, beta):
    min_size = 0
    min_require_size = 10
    N = len(dataset)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}
    
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(np.array(dataset.targets) == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    
    for i in trange(num_clients):
        np.random.shuffle(idx_batch[i])
        dict_users[i] = np.array(idx_batch[i])
    
    return dict_users

def quantity_skew(dataset, num_clients, beta):
    n_train = len(dataset)
    idxs = np.random.permutation(n_train)
    min_size = 0
    while min_size < 10:
        proportions = np.random.dirichlet(np.repeat(beta, num_clients))
        s = np.sum(proportions)
        clip_value = 10.1 * s / n_train
        proportions = np.clip(proportions, a_min = clip_value, a_max = np.max(proportions))
        proportions = proportions / proportions.sum()
        min_size = np.min(proportions * n_train)
    proportions = (np.cumsum(proportions) * n_train).astype(int)[:-1]
    batch_idxs = np.split(idxs, proportions)
    dict_users = {i: batch_idxs[i] for i in range(num_clients)}
    return dict_users
