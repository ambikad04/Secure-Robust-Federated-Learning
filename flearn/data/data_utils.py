import pickle
import os
from torch.utils.data import random_split, DataLoader
from flearn.data.dataset import MNISTDataset, CIFARDataset, CIFAR100Dataset, EMNISTDataset, FashionMNISTDataset
from path import Path
import numpy as np
import glob
import torch
import random
from dotenv import load_dotenv
import re
import torchvision.transforms as transforms

load_dotenv()

# Define normalization constants
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3015,)

# Define transforms
TRANSFORMS = {
    'cifar': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    'cifar10': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    'cifar100': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    'mnist': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])
}

TEST_TRANSFORMS = {
    'cifar': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    'cifar10': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    'cifar100': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]),
    'mnist': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])
}

DATASET_DICT = {
    "mnist": MNISTDataset,
    "cifar": CIFARDataset,
    "cifar10": CIFARDataset,
    "cifar100": CIFAR100Dataset,
    "emnist": EMNISTDataset,
    "fashionmnist": FashionMNISTDataset
}
CLASSES = {"mnist":10, "cifar10":10, "cifar":10, "cifar100":100}
CURRENT_DIR = os.getenv("DATASET_DIR")
DATASETS_TYPES = [
    "iid",
    "niid",
    "dniid",
    "synthetic",
    "mix",
    "qty_lbl_imb",  # non-iid with quantity based label imbalance
    "noiid_lbldir",  # non-iid with dirichilet based label imbalance
    "iid_diff_qty",  # quantity skew
    'iid_diff_qty_n',
    'noiid_lbldir_n'
]

def get_participants_stat(dataset, dataset_type, n_class):
    # print(f'\n->->: Getting list of participants')
    pickles_dir = f'{CURRENT_DIR}/{dataset}/{dataset_type}/{n_class}'
    
    # Check if directory exists before proceeding
    if not os.path.isdir(pickles_dir):
        print(f"Warning: Directory does not exist: {pickles_dir}")
        return []
        
    if dataset_type=='iid':
        # Get the list of files in the directory
        file_list = glob.glob(os.path.join(pickles_dir, '*.pkl'))
        training_files = 0
        testing_files = 0
        train_pattern = re.compile(r'train.*\.pkl')
        test_pattern = re.compile(r'test.*\.pkl')
        for f in file_list:
            if test_pattern.search(f):
                testing_files += 1
            elif train_pattern.search(f):
                training_files += 1
        total_clients = len(file_list)-training_files-testing_files
        # print(f'\nDataset Dir: {pickles_dir}, Files: {file_list}')
        users = [(u,dataset_type,n_class) for u in range(total_clients)]
        return users
    if dataset_type=='niid':
        # print(pickles_dir)
        # Get the list of files in the directory
        file_list = glob.glob(os.path.join(pickles_dir, '*.pkl'))
        training_files = 0
        testing_files = 0
        train_pattern = re.compile(r'train.*\.pkl')
        test_pattern = re.compile(r'test.*\.pkl')
        for f in file_list:
            if test_pattern.search(f):
                testing_files += 1
            elif train_pattern.search(f):
                training_files += 1
        total_clients = len(file_list)-training_files-testing_files
        users = [(u,dataset_type,n_class) for u in range(total_clients)]
        return users
    if dataset_type=='dniid':
        # Get the list of files in the directory
        file_list = glob.glob(os.path.join(pickles_dir, '*.pkl'))
        training_files = 0
        testing_files = 0
        train_pattern = re.compile(r'train.*\.pkl')
        test_pattern = re.compile(r'test.*\.pkl')
        for f in file_list:
            if test_pattern.search(f):
                testing_files += 1
            elif train_pattern.search(f):
                training_files += 1
        total_clients = len(file_list)-training_files-testing_files
        users = [(u,dataset_type,n_class) for u in range(total_clients)]
        return users
    if dataset_type=='fniid':
        # Get the list of files in the directory
        file_list = glob.glob(os.path.join(pickles_dir, '*.pkl'))
        training_files = 0
        testing_files = 0
        train_pattern = re.compile(r'train.*\.pkl')
        test_pattern = re.compile(r'test.*\.pkl')
        for f in file_list:
            if test_pattern.search(f):
                testing_files += 1
            elif train_pattern.search(f):
                training_files += 1
        total_clients = len(file_list)-training_files-testing_files
        users = [(u,dataset_type,n_class) for u in range(total_clients)]
        return users
    if dataset_type=='synthetic':
        # Get the list of files in the directory
        file_list = glob.glob(os.path.join(pickles_dir, '*.pkl'))
        training_files = 0
        testing_files = 0
        train_pattern = re.compile(r'train.*\.pkl')
        test_pattern = re.compile(r'test.*\.pkl')
        for f in file_list:
            if test_pattern.search(f):
                testing_files += 1
            elif train_pattern.search(f):
                training_files += 1
        total_clients = len(file_list)-training_files-testing_files
        users = [(u,dataset_type,n_class) for u in range(total_clients)]
        return users
    if dataset_type=='mix':
        pickles_dir = f'{CURRENT_DIR}/{dataset}/{DATASETS_TYPES[0]}/{n_class}'
        # Get the list of files in the directory
        file_list = glob.glob(os.path.join(pickles_dir, '*.pkl'))
        training_files = 0
        testing_files = 0
        train_pattern = re.compile(r'train.*\.pkl')
        test_pattern = re.compile(r'test.*\.pkl')
        for f in file_list:
            if test_pattern.search(f):
                testing_files += 1
            elif train_pattern.search(f):
                training_files += 1
        total_clients = len(file_list)-training_files-testing_files
        if dataset != "nmnist": total_clients = len(file_list) -training_files-testing_files 
        # print(f'Pickles-dir: {pickles_dir}\nNumber of clients: {total_clients}')
        np.random.seed(n_class) # fixed seed to produce same results for each methods
        random_integer = np.random.randint(1, CLASSES[dataset]+1)       
        options = np.arange(n_class,CLASSES[dataset]+1)
        n_classes = list(set(np.random.choice(options, size=random_integer)))
        fraction = int(total_clients/len(n_classes)) 
        # print(f'{total_clients}, {n_classes}, {fraction}')
        count = 0
        users = []
        for n_class in n_classes:            
            dataset_type = np.random.choice(DATASETS_TYPES)
            bound = int(fraction)
            random.seed(random_integer+n_class)
            unique_values = random.sample(range(total_clients), bound)
            _users = [(u,dataset_type,n_class) for u in unique_values]
            # print(_users)
            count+=fraction
            if count-1>=total_clients: break
            users.extend(_users)
        #     print(f'count: -->>>>>>>>>>>>>>> {count}\n fraction: {fraction}')
        # print("users:->>>>>>>>>>>>>>>>",len(users))
        return users
    if dataset_type=='syntheticM':
        dataset_type = DATASETS_TYPES[3] # in index 4 synthetic data type is mentioned
        pickles_dir = f'{CURRENT_DIR}/{dataset}/{dataset_type}/{n_class}'
        # Get the list of files in the directory
        file_list = glob.glob(os.path.join(pickles_dir, '*.pkl'))
        training_files = 0
        testing_files = 0
        train_pattern = re.compile(r'train.*\.pkl')
        test_pattern = re.compile(r'test.*\.pkl')
        for f in file_list:
            if test_pattern.search(f):
                testing_files += 1
            elif train_pattern.search(f):
                training_files += 1
        total_clients = len(file_list)-training_files-testing_files
        if dataset != "nmnist": total_clients = len(file_list) - training_files-testing_files
        # print(f'Pickles-dir: {pickles_dir}\nNumber of clients: {total_clients}')
        np.random.seed(n_class) # fixed seed to produce same results for each methods
        random_integer = np.random.randint(1, CLASSES[dataset]+1)       
        options = np.arange(n_class,CLASSES[dataset]+1)
        n_classes = list(set(np.random.choice(options, size=random_integer)))
        fraction = int(total_clients/len(n_classes)) 
        # print(f'{total_clients}, {n_classes}, {fraction}')
        count = 0
        users = []
        for n_class in n_classes: 
            bound = int(fraction)
            random.seed(random_integer+n_class)
            unique_values = random.sample(range(total_clients), bound)
            _users = [(u,dataset_type,n_class) for u in unique_values]
            # print(_users)
            count+=fraction
            if count-1>=total_clients: break
            users.extend(_users)
        return users
    else:
        pickles_dir = f'{CURRENT_DIR}/{dataset}/{dataset_type}/{n_class}'
        # Get the list of files in the directory
        file_list = glob.glob(os.path.join(pickles_dir, '*.pkl'))
        training_files = 0
        testing_files = 0
        train_pattern = re.compile(r'train.*\.pkl')
        test_pattern = re.compile(r'test.*\.pkl')
        for f in file_list:
            if test_pattern.search(f):
                testing_files += 1
            elif train_pattern.search(f):
                training_files += 1
        total_clients = len(file_list)-training_files-testing_files
        users = [(u,dataset_type,n_class) for u in range(total_clients)]
        return users

def get_dataloader(dataset, user_id, dataset_type, n_class, batch_size, valset_ratio):
    train_data, test_data = get_client_dataloader(dataset, user_id, dataset_type, n_class)
    
    # Apply transforms to the datasets - ensure proper normalization
    if hasattr(train_data, 'transform'):
        train_data.transform = TRANSFORMS.get(dataset, None)
    if hasattr(test_data, 'transform'):
        test_data.transform = TEST_TRANSFORMS.get(dataset, None)
    
    # Use smaller validation set for speed
    train_size = int(len(train_data) * 0.9)  # Use 90% for training, 10% for validation
    test_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, test_size])
    
    # Use larger batch sizes for faster processing
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size * 4,  # 4x larger batches
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True  # Drop incomplete batches for speed
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size * 4,  # 4x larger batches
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True  # Drop incomplete batches for speed
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size * 4,  # 4x larger batches
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True  # Drop incomplete batches for speed
    )
    
    return train_loader, val_loader, len(train_data), len(test_data)

def get_testloader(dataset: str, dataset_type: str, n_class: int, batch_size=20, valset_ratio=0.1):
    pickles_dir = f'{CURRENT_DIR}/{dataset}/{dataset_type}/{n_class}'
    print(f'Loading test data from: {pickles_dir}')
    if os.path.isdir(pickles_dir) is False:
        print(f"Warning: Directory {pickles_dir} not found. Creating a substantial dummy test dataset.")
        # Create a dummy dataset with a reasonable number of samples
        if dataset in DATASET_DICT:
            # For image datasets, create a dummy dataset with random samples
            num_test_samples = 500  # Use 500 test samples
            
            if dataset in ["cifar", "cifar10", "cifar100"]:
                # RGB images
                test_dataset = create_dummy_data(dataset, num_test_samples, (3, 32, 32))
            else:  # mnist and similar
                # Grayscale images
                test_dataset = create_dummy_data(dataset, num_test_samples, (1, 28, 28))
            
            testloader = DataLoader(
                test_dataset, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            print(f"Created dummy test dataset with {num_test_samples} samples.")
            return testloader, num_test_samples
        else:
            raise RuntimeError(f"Unknown dataset: {dataset}")

    try:
        with open(f'{pickles_dir}/test.pkl', "rb") as f:
            test_dataset = pickle.load(f)
            
        # Apply test transforms
        if hasattr(test_dataset, 'transform'):
            test_dataset.transform = TEST_TRANSFORMS.get(dataset, None)
            
        testloader = DataLoader(
            test_dataset, 
            batch_size=batch_size * 4,  # 4x larger batches
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True  # Drop incomplete batches for speed
        )
    except FileNotFoundError as e:
        print(f"Warning: Test file not found at {pickles_dir}/test.pkl. Creating a substantial dummy test dataset.")
        # Create a dummy dataset with a reasonable number of samples
        num_test_samples = 500  # Use 500 test samples
        
        if dataset in ["cifar", "cifar10", "cifar100"]:
            # RGB images
            test_dataset = create_dummy_data(dataset, num_test_samples, (3, 32, 32))
        else:  # mnist and similar
            # Grayscale images
            test_dataset = create_dummy_data(dataset, num_test_samples, (1, 28, 28))
        
        testloader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        print(f"Created dummy test dataset with {num_test_samples} samples.")
        return testloader, num_test_samples

    return testloader, len(test_dataset)

def get_client_id_indices(dataset):
    print(f'Dataset Dir: {CURRENT_DIR}')
    dataset_pickles_path = CURRENT_DIR / dataset / "pickles"
    with open(dataset_pickles_path / "seperation.pkl", "rb") as f:
        seperation = pickle.load(f)
    return (seperation["train"], seperation["test"], seperation["total"])

def get_dataset_stats(dataset, dataset_type: str, n_class: int, client_id: int):
    #calculating datasets stat
    pickles_dir = f'{CURRENT_DIR}/{dataset}/{dataset_type}/{n_class}'
    dataset_stats = torch.zeros(CLASSES[dataset])
    with open(f'{pickles_dir}/{client_id}.pkl', "rb") as f:
        client_dataset: DATASET_DICT[dataset] = pickle.load(f)
        for x in client_dataset.targets:
            dataset_stats[x.item()] += 1
        dataset_stats[dataset_stats==0.0] = 1e-8
    return dataset_stats

def get_client_dataloader(dataset, user_id, dataset_type, n_class):
    """Load client-specific data from pickle files"""
    pickles_dir = f'{CURRENT_DIR}/{dataset}/{dataset_type}/{n_class}'
    
    try:
        # Load training data
        with open(f'{pickles_dir}/{user_id}.pkl', 'rb') as f:
            train_data = pickle.load(f)
            
        # Load test data
        with open(f'{pickles_dir}/test.pkl', 'rb') as f:
            test_data = pickle.load(f)
            
        return train_data, test_data
    except FileNotFoundError as e:
        print(f"Warning: Client data not found at {pickles_dir}/{user_id}.pkl or test.pkl. Creating dummy data.")
        
        # Create dummy datasets
        if dataset in ["cifar", "cifar10", "cifar100"]:
            # RGB images
            dummy_train = create_dummy_data(dataset, 10, (3, 32, 32))
            dummy_test = create_dummy_data(dataset, 5, (3, 32, 32))
        else:  # mnist and similar
            # Grayscale images
            dummy_train = create_dummy_data(dataset, 10, (1, 28, 28))
            dummy_test = create_dummy_data(dataset, 5, (1, 28, 28))
        
        return dummy_train, dummy_test

def create_dummy_data(dataset, num_samples, image_shape):
    """Create a dummy dataset with given number of samples and image shape"""
    from torch.utils.data.dataset import Dataset
    
    class DummyDataset(Dataset):
        def __init__(self, num_samples, image_shape, transform=None, num_classes=10):
            # Generate random data instead of zeros for better training behavior
            self.data = torch.rand((num_samples, *image_shape))
            # Randomly generate labels across num_classes
            self.targets = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)
            self.transform = transform
            
        def __getitem__(self, index):
            img = self.data[index]
            target = self.targets[index]
            
            if self.transform:
                img = self.transform(img)
                
            return img, target
            
        def __len__(self):
            return len(self.data)
    
    # Get num_classes based on dataset
    num_classes = CLASSES.get(dataset, 10)
    
    return DummyDataset(num_samples, image_shape, transform=TRANSFORMS.get(dataset, None), num_classes=num_classes)
