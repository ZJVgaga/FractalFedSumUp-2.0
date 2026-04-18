import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
import random
import numpy as np
import io
import pickle
import pandas as pd
from sklearn.utils import resample, shuffle
import requests
from tqdm import tqdm
from PIL import Image
try:
    from medmnist.info import INFO
except ImportError:
    INFO = {}

def test_indices(target_train_indices, client_indices):
    """
    Check if target_train_indices equals the union of client_indices.
    
    Parameters:
    - target_train_indices: A list or numpy array containing all training sample indices.
    - client_indices: A list containing multiple sublists, each representing the training indices held by a client.
    
    Returns:
    - A tuple containing two boolean values:
        - The first boolean indicates whether target_train_indices completely covers all indices in client_indices.
        - The second boolean indicates whether the union of client_indices completely covers all indices in target_train_indices.
    """
    # Convert target_train_indices to a set for faster lookup
    target_set = set(target_train_indices)
    
    # Convert client_indices to the union of sets
    client_set = set().union(*client_indices)
    
    # Check if target_train_indices completely covers all indices in client_indices
    all_client_indices_in_target = client_set.issubset(target_set)
    
    # Check if the union of client_indices completely covers all indices in target_train_indices
    all_target_indices_in_clients = target_set.issubset(client_set)
    
    # Print missing indices (if any)
    if not all_client_indices_in_target:
        missing_indices = [idx for idx in client_set if idx not in target_set]
        print(f"Missing indices in target_train_indices: {missing_indices}")
    
    if not all_target_indices_in_clients:
        missing_indices = [idx for idx in target_set if idx not in client_set]
        print(f"Missing indices in client_indices: {missing_indices}")
    
    return all_client_indices_in_target, all_target_indices_in_clients

# Used to measure communication overhead
def measure_size(obj):
    buffer = io.BytesIO()
    pickle.dump(obj, buffer)
    return buffer.getbuffer().nbytes

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def split_datasets(trainset, testset, config, target_ratio=0.5):
    # Get all indices of the training set and test set
    train_indices = list(range(len(trainset)))
    test_indices = list(range(len(testset)))
    
    # Randomly shuffle indices
    np.random.seed(config.get("seed"))
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # Split indices based on target_ratio
    target_train_size = int(len(train_indices) * target_ratio)
    target_test_size = int(len(test_indices) * target_ratio)

    # Split training set indices
    target_train_indices = train_indices[:target_train_size]
    shadow_train_indices = train_indices[target_train_size:]

    # Split test set indices
    target_test_indices = test_indices[:target_test_size]
    shadow_test_indices = test_indices[target_test_size:]

    return target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices

def get_dataset_indices(trainset, testset, config):

    if config.get("attack_mode") == "MIA":
        # Split the dataset
        target_ratio = config.get("MIA_data_target_ratio")
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = split_datasets(trainset, testset, config, target_ratio)

        # Create DataLoader
       
        return target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices
    else:
    
        # Get all indices of the training set and test set
        train_indices = list(range(len(trainset)))
        test_indices = list(range(len(testset)))

        return train_indices, None, test_indices, None

def partition_dataset_for_clients(data_info, trainset, target_train_indices, alpha, config):
    # Set random seed to ensure reproducible results
    np.random.seed(config.get("seed"))
    
    # Initialize minimum dataset size to 0, used to control the loop
    min_size = 0
    # Set the minimum required data amount per client
    min_require_size = 10
    
    # Get the number of classes K
    K = data_info["num_classes"]
    
    # Get the number of clients
    client_num = config.get("client_num")
    
    # Extract labels corresponding to the target training indices
    labels = np.array(trainset.targets, dtype='int64')[target_train_indices]
    
    # Total number of samples N
    N = labels.shape[0]
    
    print(f"partition_dataset_for_clients: Starting data partitioning, alpha={alpha}, client_num={client_num}, K={K}, N={N}")
    
    # Initialize dictionary to store data indices for each client
    dict_users = {}
    # Initialize dictionary to store classes contained in each client
    dict_classes = {}

    # Loop until each client has at least min_require_size samples
    iteration = 0
    while min_size < min_require_size:
        iteration += 1
        print(f"partition_dataset_for_clients: Iteration {iteration}, current min_size={min_size}")
        
        # Initialize data index list for each client
        idx_batch = [[] for _ in range(client_num)]
        
        # Process each class k
        for k in range(K):
            # Find all sample indices with label k
            idx_label = np.where(labels == k)[0]
            idx_label_list = idx_label.tolist()
            #print("idx_label:", idx_label_list)
            # Map these indices back to indices in the original dataset
            # Assume idx_label_list is already the list you want to use to index target_train_indices
            idx_k = [target_train_indices[i] for i in idx_label_list]
            # Randomly shuffle these indices
            np.random.shuffle(idx_k)
            
            # Use Dirichlet distribution to generate data allocation proportions for each client
            proportions = np.random.dirichlet(np.repeat(alpha, client_num))
            # Adjust proportions to prevent some clients from getting too much data
            proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
            # Normalize the adjusted proportions
            proportions = proportions / proportions.sum()
            # Calculate split points
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # Split idx_k to each client according to the calculated proportions
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        
        # Update min_size to the current minimum client data amount
        min_size = min([len(idx_j) for idx_j in idx_batch])
        
        # Prevent infinite loop
        if iteration > 100:
            print(f"partition_dataset_for_clients: Warning: Reached maximum iteration count 100, forcing loop exit")
            print(f"partition_dataset_for_clients: Current data amount per client: {[len(idx_j) for idx_j in idx_batch]}")
            break

    # Randomly shuffle data indices for each client
    for j in range(client_num):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]

    # Initialize class count for each client
    net_cls_counts = {}

    # Create index mapping dictionary to optimize lookup performance
    index_map = {idx: i for i, idx in enumerate(target_train_indices)}
    
    # Process data indices for each client, count the number of each class in each client
    for net_i, dataidx in dict_users.items():
        dict_classes[net_i] = []

        # Used to store original dataset labels corresponding to all samples of the current client
        labels_for_client = []
        
        # Traverse data indices for each client
        for idx in dataidx:
            # Use dictionary for fast index lookup
            original_idx_position = index_map.get(idx)
            if original_idx_position is not None:
                # Use this position to get the corresponding real sample label
                label = labels[original_idx_position]
                labels_for_client.append(label)
            else:
                print(f"Warning: Index {idx} is not in target_train_indices")

        # Convert list to numpy array for subsequent processing
        if labels_for_client:
            labels_for_client = np.array(labels_for_client)
        
            # Use np.unique to calculate occurrence count of each class
            unq, unq_cnt = np.unique(labels_for_client, return_counts=True)
        
            # Create a dictionary where keys are classes and values are sample counts for that class
            tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        
            # Store class count for each client
            net_cls_counts[net_i] = tmp
        
            # If sample count for a class is >= 10, add it to dict_classes
            for c, cnt in tmp.items():
                dict_classes[net_i].append(int(c))
        else:
            net_cls_counts[net_i] = {}
            print(f"Warning: Client {net_i} has no valid data")

    print('partition_dataset: Data statistics: %s' % str(net_cls_counts))

    # Build dictionary to return
    data_dict = {
        "client_idx": [dict_users[i] for i in range(client_num)],
        "client_classes": [dict_classes[i] for i in range(client_num)],
    }
    client_indices, client_classes = data_dict['client_idx'], data_dict['client_classes']
    
    # Return results
    return client_indices, client_classes

def get_dataset(dataset, data_path="./data", hyperparameter_experiment=False):
    """
    Load dataset
    
    Parameters:
    - dataset: Dataset name
    - data_path: Data storage path
    - hyperparameter_experiment: Whether it's a hyperparameter experiment, if True, split 10000 images from training set as validation set
    
    Returns:
    - dataset_info: Dataset information dictionary
    - dst_train: Training set (if hyperparameter_experiment=True, it's the original training set minus 10000 images)
    - dst_test: Test set (if hyperparameter_experiment=True, it's the 10000 validation set split from training set)
    """
    print(f"\n===== Starting to load dataset: {dataset} =====")  # Overall progress prompt
    if hyperparameter_experiment:
        print(f"[Hyperparameter Experiment] Will perform validation set split: split 10000 images from training set as validation set")
    
    if dataset == 'MNIST':
        print(f"[MNIST] Starting processing...")
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.1307] * 3
        std = [0.3081] * 3
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=mean, std=std)
        ])
        print(f"[MNIST] Loading training set...")
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        print(f"[MNIST] Loading test set...")
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
        print(f"[MNIST] Loading completed, training set sample count: {len(dst_train)}, test set sample count: {len(dst_test)}")

    elif dataset == 'FashionMNIST':
        print(f"[FashionMNIST] Starting processing...")
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.2861] * 3
        std = [0.3530] * 3
        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=mean, std=std)
        ])
        print(f"[FashionMNIST] Loading training set...")
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
        print(f"[FashionMNIST] Loading test set...")
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        print(f"[FashionMNIST] Loading completed, training set sample count: {len(dst_train)}, test set sample count: {len(dst_test)}")

    elif dataset == 'SVHN':
        print(f"[SVHN] Starting processing...")
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        print(f"[SVHN] Loading training set...")
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)
        print(f"[SVHN] Loading test set...")
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
        print(f"[SVHN] Loading completed, training set sample count: {len(dst_train)}, test set sample count: {len(dst_test)}")

    elif dataset == 'CIFAR10':
        print(f"[CIFAR10] Starting processing...")
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        print(f"[CIFAR10] Loading training set...")
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        print(f"[CIFAR10] Loading test set...")
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        print(f"[CIFAR10] Loading completed, training set sample count: {len(dst_train)}, test set sample count: {len(dst_test)}")

    elif dataset == 'CIFAR100':
        print(f"[CIFAR100] Starting processing...")
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        print(f"[CIFAR100] Loading training set...")
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
        print(f"[CIFAR100] Loading test set...")
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        print(f"[CIFAR100] Loading completed, training set sample count: {len(dst_train)}, test set sample count: {len(dst_test)}")

    elif dataset == 'COVIDx':
        print(f"[COVIDx] Starting processing...")
        channel = 3
        im_size = (64, 64)
        num_classes = 2
        mean = [0.5] * 3
        std = [0.5] * 3
        class_names = ['Negative', 'Positive']
        rawdata_path = os.path.join(data_path, 'COVIDx', 'rawdata')
        print(f"[COVIDx] Checking data path: {rawdata_path}")
        if not os.path.exists(rawdata_path):
            raise FileNotFoundError(f"COVIDx rawdata not found at {rawdata_path}")

        def load_csv(csv_path, image_folder):
            print(f"[COVIDx] Loading CSV file: {csv_path}")
            df = pd.read_csv(csv_path, sep=" ", header=None)
            df.columns = ['patient_id', 'file_name', 'class', 'data_source']
            df['class'] = (df['class'] == 'positive').astype(int)
            return ImageDataset(dataframe=df, image_folder=image_folder, transform=None)

        print(f"[COVIDx] Loading training set CSV...")
        train_ds = load_csv(os.path.join(rawdata_path, 'train.txt'), os.path.join(rawdata_path, 'train/'))
        print(f"[COVIDx] Loading validation set CSV...")
        val_ds = load_csv(os.path.join(rawdata_path, 'val.txt'), os.path.join(rawdata_path, 'val/'))
        print(f"[COVIDx] Loading test set CSV...")
        test_ds = load_csv(os.path.join(rawdata_path, 'test.txt'), os.path.join(rawdata_path, 'test/'))

        print(f"[COVIDx] Merging training set and validation set...")
        train_val_df = pd.concat([train_ds