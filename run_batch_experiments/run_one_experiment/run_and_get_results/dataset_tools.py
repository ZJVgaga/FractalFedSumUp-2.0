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
    检查 target_train_indices 是否等于 client_indices 的并集。
    
    参数:
    - target_train_indices: 一个包含所有训练样本索引的列表或 numpy 数组。
    - client_indices: 一个包含多个子列表的列表，每个子列表代表一个客户端持有的训练索引。
    
    返回:
    - 包含两个布尔值的元组：
        - 第一个布尔值表示 target_train_indices 是否完全覆盖了 client_indices 的所有索引。
        - 第二个布尔值表示 client_indices 的并集是否完全覆盖了 target_train_indices 的所有索引。
    """
    # 将 target_train_indices 转换为集合以加快查找速度
    target_set = set(target_train_indices)
    
    # 将 client_indices 转换为集合的并集
    client_set = set().union(*client_indices)
    
    # 检查 target_train_indices 是否完全覆盖了 client_indices 的所有索引
    all_client_indices_in_target = client_set.issubset(target_set)
    
    # 检查 client_indices 的并集是否完全覆盖了 target_train_indices 的所有索引
    all_target_indices_in_clients = target_set.issubset(client_set)
    
    # 打印缺失的索引（如果有的话）
    if not all_client_indices_in_target:
        missing_indices = [idx for idx in client_set if idx not in target_set]
        print(f"Missing indices in target_train_indices: {missing_indices}")
    
    if not all_target_indices_in_clients:
        missing_indices = [idx for idx in target_set if idx not in client_set]
        print(f"Missing indices in client_indices: {missing_indices}")
    
    return all_client_indices_in_target, all_target_indices_in_clients
#用于测量通讯开销的
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

def split_datasets(trainset, testset,config, target_ratio=0.5):
    # 获取训练集和测试集的所有索引
    train_indices = list(range(len(trainset)))
    test_indices = list(range(len(testset)))
    
    # 随机打乱索引
    np.random.seed(config.get("seed"))
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # 根据target_ratio分割索引
    target_train_size = int(len(train_indices) * target_ratio)
    target_test_size = int(len(test_indices) * target_ratio)

    # 分割训练集索引
    target_train_indices = train_indices[:target_train_size]
    shadow_train_indices = train_indices[target_train_size:]

    # 分割测试集索引
    target_test_indices = test_indices[:target_test_size]
    shadow_test_indices = test_indices[target_test_size:]

    return target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices

def get_dataset_indices(trainset, testset,config):

    if config.get("attack_mode")=="MIA":
    # 分割数据集
        target_ratio=config.get("MIA_data_target_ratio")
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices= split_datasets(trainset, testset,config, target_ratio)

    # 创建DataLoader
       
        return target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices
    else:
    
        # 获取训练集和测试集的所有索引
        train_indices = list(range(len(trainset)))
        test_indices = list(range(len(testset)))

        return train_indices,None, test_indices,None

def partition_dataset_for_clients(data_info, trainset, target_train_indices, alpha,config):
    # 设置随机种子以确保结果可复现
    np.random.seed(config.get("seed"))
    
    # 初始化最小数据集大小为0，用于控制循环
    min_size = 0
    # 设定每个客户端需要的最小数据量
    min_require_size = 10
    
    # 获取类别数量K
    K = data_info["num_classes"]
    
    # 获取客户端数量
    client_num = config.get("client_num")
    
    # 提取目标训练索引对应的标签
    labels = np.array(trainset.targets, dtype='int64')[target_train_indices]
    
    # 样本总数N
    N = labels.shape[0]
    
    print(f"partition_dataset_for_clients: 开始数据划分，alpha={alpha}, client_num={client_num}, K={K}, N={N}")
    
    # 初始化字典，用于存储每个客户端的数据索引
    dict_users = {}
    # 初始化字典，用于存储每个客户端包含的类别
    dict_classes = {}

    # 循环直到每个客户端至少有min_require_size个样本
    iteration = 0
    while min_size < min_require_size:
        iteration += 1
        print(f"partition_dataset_for_clients: 第{iteration}次迭代，当前min_size={min_size}")
        
        # 初始化每个客户端的数据索引列表
        idx_batch = [[] for _ in range(client_num)]
        
        # 对每个类别k进行处理
        for k in range(K):
            # 找到所有标签为k的样本索引
            idx_label = np.where(labels == k)[0]
            idx_label_list = idx_label.tolist()
            #print("idx_label:",idx_label_list)
            # 将这些索引映射回原始数据集中的索引
            # 假设 idx_label_list 已经是你想用来索引 target_train_indices 的列表
            idx_k = [target_train_indices[i] for i in idx_label_list]
            # 随机打乱这些索引
            np.random.shuffle(idx_k)
            
            # 使用Dirichlet分布生成每个客户端的数据分配比例
            proportions = np.random.dirichlet(np.repeat(alpha, client_num))
            # 调整比例，防止某些客户端获得过多的数据
            proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
            # 归一化调整后的比例
            proportions = proportions / proportions.sum()
            # 计算分割点
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            # 根据计算出的比例将idx_k分割给各个客户端
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        
        # 更新min_size为当前最小客户端数据量
        min_size = min([len(idx_j) for idx_j in idx_batch])
        
        # 防止无限循环
        if iteration > 100:
            print(f"partition_dataset_for_clients: 警告：已达到最大迭代次数100，强制退出循环")
            print(f"partition_dataset_for_clients: 当前各客户端数据量: {[len(idx_j) for idx_j in idx_batch]}")
            break

    # 对每个客户端的数据索引进行随机打乱
    for j in range(client_num):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]

    # 初始化每个客户端的类别计数
    net_cls_counts = {}

    # 创建索引映射字典，优化查找性能
    index_map = {idx: i for i, idx in enumerate(target_train_indices)}
    
    # 对每个客户端的数据索引进行处理，统计每个客户端中各类别的数量
    for net_i, dataidx in dict_users.items():
        dict_classes[net_i] = []

        # 用于存储当前客户端所有样本对应的原始数据集标签
        labels_for_client = []
        
        # 遍历每个客户端的数据索引
        for idx in dataidx:
            # 使用字典快速查找索引位置
            original_idx_position = index_map.get(idx)
            if original_idx_position is not None:
                # 使用这个位置获取对应的真实样本标签
                label = labels[original_idx_position]
                labels_for_client.append(label)
            else:
                print(f"警告：索引 {idx} 不在 target_train_indices 中")

        # 将列表转换为numpy数组以便后续处理
        if labels_for_client:
            labels_for_client = np.array(labels_for_client)
        
            # 使用np.unique计算每个类别的出现次数
            unq, unq_cnt = np.unique(labels_for_client, return_counts=True)
        
            # 创建一个字典，键是类别，值是该类别的样本数
            tmp = {int(unq[i]): int(unq_cnt[i]) for i in range(len(unq))}
        
            # 存储每个客户端的类别计数
            net_cls_counts[net_i] = tmp
        
            # 如果某个类别的样本数量大于等于10，则将其添加到dict_classes
            for c, cnt in tmp.items():
                dict_classes[net_i].append(int(c))
        else:
            net_cls_counts[net_i] = {}
            print(f"警告：客户端 {net_i} 没有有效数据")

    print('partition_dataset:Data statistics: %s' % str(net_cls_counts))

    # 构建要返回的字典
    data_dict = {
        "client_idx": [dict_users[i] for i in range(client_num)],
        "client_classes": [dict_classes[i] for i in range(client_num)],
    }
    client_indices, client_classes = data_dict['client_idx'], data_dict['client_classes']
    
    # 返回结果
    return client_indices, client_classes





def get_dataset(dataset, data_path="./data", hyperparameter_experiment=False):
    """
    加载数据集
    
    参数:
    - dataset: 数据集名称
    - data_path: 数据存储路径
    - hyperparameter_experiment: 是否为超参数实验，如果是则从训练集中分出10000张作为验证集
    
    返回:
    - dataset_info: 数据集信息字典
    - dst_train: 训练集（如果hyperparameter_experiment=True，则为原始训练集减去10000张）
    - dst_test: 测试集（如果hyperparameter_experiment=True，则为从训练集中分出的10000张验证集）
    """
    print(f"\n===== 开始加载数据集: {dataset} =====")  # 总进度提示
    if hyperparameter_experiment:
        print(f"[超参数实验] 将进行验证集分割：从训练集中分出10000张作为验证集")
    
    if dataset == 'MNIST':
        print(f"[MNIST] 开始处理...")
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
        print(f"[MNIST] 加载训练集...")
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        print(f"[MNIST] 加载测试集...")
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
        print(f"[MNIST] 加载完成，训练集样本数: {len(dst_train)}, 测试集样本数: {len(dst_test)}")

    elif dataset == 'FashionMNIST':
        print(f"[FashionMNIST] 开始处理...")
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
        print(f"[FashionMNIST] 加载训练集...")
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
        print(f"[FashionMNIST] 加载测试集...")
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        print(f"[FashionMNIST] 加载完成，训练集样本数: {len(dst_train)}, 测试集样本数: {len(dst_test)}")

    elif dataset == 'SVHN':
        print(f"[SVHN] 开始处理...")
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        print(f"[SVHN] 加载训练集...")
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)
        print(f"[SVHN] 加载测试集...")
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
        print(f"[SVHN] 加载完成，训练集样本数: {len(dst_train)}, 测试集样本数: {len(dst_test)}")

    elif dataset == 'CIFAR10':
        print(f"[CIFAR10] 开始处理...")
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        print(f"[CIFAR10] 加载训练集...")
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        print(f"[CIFAR10] 加载测试集...")
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        print(f"[CIFAR10] 加载完成，训练集样本数: {len(dst_train)}, 测试集样本数: {len(dst_test)}")

    elif dataset == 'CIFAR100':
        print(f"[CIFAR100] 开始处理...")
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        print(f"[CIFAR100] 加载训练集...")
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
        print(f"[CIFAR100] 加载测试集...")
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        print(f"[CIFAR100] 加载完成，训练集样本数: {len(dst_train)}, 测试集样本数: {len(dst_test)}")

    elif dataset == 'COVIDx':
        print(f"[COVIDx] 开始处理...")
        channel = 3
        im_size = (64, 64)
        num_classes = 2
        mean = [0.5] * 3
        std = [0.5] * 3
        class_names = ['Negative', 'Positive']
        rawdata_path = os.path.join(data_path, 'COVIDx', 'rawdata')
        print(f"[COVIDx] 检查数据路径: {rawdata_path}")
        if not os.path.exists(rawdata_path):
            raise FileNotFoundError(f"COVIDx rawdata not found at {rawdata_path}")

        def load_csv(csv_path, image_folder):
            print(f"[COVIDx] 加载CSV文件: {csv_path}")
            df = pd.read_csv(csv_path, sep=" ", header=None)
            df.columns = ['patient_id', 'file_name', 'class', 'data_source']
            df['class'] = (df['class'] == 'positive').astype(int)
            return ImageDataset(dataframe=df, image_folder=image_folder, transform=None)

        print(f"[COVIDx] 加载训练集CSV...")
        train_ds = load_csv(os.path.join(rawdata_path, 'train.txt'), os.path.join(rawdata_path, 'train/'))
        print(f"[COVIDx] 加载验证集CSV...")
        val_ds = load_csv(os.path.join(rawdata_path, 'val.txt'), os.path.join(rawdata_path, 'val/'))
        print(f"[COVIDx] 加载测试集CSV...")
        test_ds = load_csv(os.path.join(rawdata_path, 'test.txt'), os.path.join(rawdata_path, 'test/'))

        print(f"[COVIDx] 合并训练集和验证集...")
        train_val_df = pd.concat([train_ds.dataframe, val_ds.dataframe], ignore_index=True)
        negative = train_val_df[train_val_df['class'] == 0]
        positive = train_val_df[train_val_df['class'] == 1]
        min_samples = min(len(negative), len(positive))
        print(f"[COVIDx] 平衡类别（负样本: {len(negative)}, 正样本: {len(positive)}, 目标: {min_samples}）...")
        positive_downsampled = resample(positive, replace=True, n_samples=min_samples, random_state=1)
        train_val_balanced = pd.concat([negative, positive_downsampled], ignore_index=True)
        train_val_balanced = shuffle(train_val_balanced, random_state=1)

        transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=mean, std=std)
        ])

        print(f"[COVIDx] 创建训练集数据集...")
        dst_train = ImageDataset(
            dataframe=train_val_balanced,
            image_folder=os.path.join(rawdata_path, 'train/'),
            transform=transform
        )
        print(f"[COVIDx] 创建测试集数据集...")
        dst_test = ImageDataset(
            dataframe=test_ds.dataframe,
            image_folder=os.path.join(rawdata_path, 'test/'),
            transform=transform
        )
        print(f"[COVIDx] 加载完成，训练集样本数: {len(dst_train)}, 测试集样本数: {len(dst_test)}")

    elif dataset == 'TinyImageNet':
        print(f"[TinyImageNet] 开始处理...")
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data_path_tiny = os.path.join(data_path, 'tinyimagenet.pt')
        print(f"[TinyImageNet] 加载数据文件: {data_path_tiny}")
        data = torch.load(data_path_tiny, map_location='cpu')
        class_names = data['classes']

        print(f"[TinyImageNet] 处理训练集图像...")
        images_train = data['images_train'].detach().float() / 255.0
        labels_train = data['labels_train'].detach()
        for c in range(channel):
            images_train[:, c] = (images_train[:, c] - mean[c]) / std[c]
        dst_train = TensorDataset(images_train, labels_train)

        print(f"[TinyImageNet] 处理验证集图像...")
        images_val = data['images_val'].detach().float() / 255.0
        labels_val = data['labels_val'].detach()
        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]
        dst_test = TensorDataset(images_val, labels_val)
        print(f"[TinyImageNet] 加载完成，训练集样本数: {len(dst_train)}, 测试集样本数: {len(dst_test)}")

# ---------------------- MedMNIST分支（核心修改部分） ----------------------
    elif dataset.startswith("MedMNIST"):
        print(f"[MedMNIST] 开始处理...")

 
        data_path = os.path.abspath(data_path)
        print(f"[MedMNIST] 修复：强制root绝对路径={data_path}")
 
        def download_with_progress(url, save_path, progress_bar=True):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024
                progress = tqdm(total=total_size, unit='iB', unit_scale=True, disable=not progress_bar)
                with open(save_path, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress.update(len(data))
                        f.write(data)
                progress.close()
                if total_size != 0 and progress.n != total_size:
                    raise RuntimeError("下载文件损坏或不完整")
                print(f"[MedMNIST] 下载完成: {save_path}")
            except requests.exceptions.RequestException as e:
                print(f"[MedMNIST] 警告: 无法从 {url} 下载数据集: {e}")
                print(f"[MedMNIST] 尝试使用本地缓存文件（如果存在）...")
                if not os.path.exists(save_path):
                    raise RuntimeError(f"无法下载数据集且本地缓存文件不存在: {save_path}")
                else:
                    print(f"[MedMNIST] 使用本地缓存文件: {save_path}")
 
        parts = dataset.split("-")
        if len(parts) < 2:
            raise ValueError("MedMNIST格式应为：MedMNIST-<子数据集名称>-<尺寸>")
        subset = parts[1].lower()
        # ---------------------- 修改1：强制目标尺寸为32 ----------------------
        target_size = 32  # 固定尺寸为32，忽略输入的尺寸参数
        print(f"[MedMNIST] 子数据集: {subset}, 目标尺寸: {target_size}×{target_size}")
 
        info = INFO[subset]
        # ---------------------- 修改2：兼容字符串和字典格式的URL（核心修复） ----------------------
        # 检查info["url"]是否为字典（支持多尺寸）
        if isinstance(info["url"], dict):
            # 字典格式：优先获取target_size，否则用28
            url = info["url"].get(f"{target_size}", info["url"].get("28"))
        else:
            # 字符串格式：直接使用URL（默认28尺寸）
            url = info["url"]
            print(f"[提示] 数据集{subset}仅支持默认尺寸，使用URL: {url}")
        
        # 若获取的URL为28尺寸且目标尺寸非28，提示后续Resize
        if "28" in url and target_size != 28:
            print(f"[警告] 原始数据集为28×28，将通过Resize缩放至{target_size}×{target_size}")
        print(f"[MedMNIST] 调试：数据集URL={url}")
 
        # ---------------------- 修改3：文件名标注尺寸，避免与原28版本混淆 ----------------------
        save_dir = os.path.join(data_path, "medmnist")
        save_path = os.path.join(save_dir, f"{subset}.npz")  # 新增尺寸后缀
        print(f"[MedMNIST] 调试：本地文件路径={save_path}")
        print(f"[MedMNIST] 调试：文件是否存在={os.path.exists(save_path)}")
 
        if not os.path.exists(save_path):
            print(f"[MedMNIST] 下载数据集: {url}")
            download_with_progress(url, save_path)
        else:
            print(f"[MedMNIST] 数据集已存在: {save_path}")
 
        # 动态标签处理（保持不变）
        print(f"[MedMNIST] 手动加载数据文件: {save_path}")
        data = np.load(save_path)
        train_images = data['train_images']
        train_labels = data['train_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
 
        def process_labels(labels):
            labels = labels.squeeze()
            if labels.ndim == 2:
                return labels[:, 0]
            elif labels.ndim == 1:
                return labels
            else:
                raise ValueError(f"不支持的标签维度: {labels.ndim}")
 
        train_labels = process_labels(train_labels)
        test_labels = process_labels(test_labels)
        print(f"[MedMNIST] 手动加载完成：训练集{train_images.shape}，测试集{test_images.shape}")
        print(f"[MedMNIST] 调试：训练集标签形状={train_labels.shape}（处理后单标签格式）")

	# ==============================================
    # 核心修改：裁剪训练集至60000张，测试集至10000张
    # ==============================================
    # 训练集裁剪（最多60000张，不足则全部保留）
        target_train_size = 60000
        if len(train_images) > target_train_size:
         # 随机裁剪（固定种子确保可复现）
            np.random.seed(42)  # 固定随机种子
            train_indices = np.random.choice(len(train_images), size=target_train_size, replace=False)
            train_images = train_images[train_indices]
            train_labels = train_labels[train_indices]
            print(f"[裁剪] 训练集从{len(train_images)+len(train_indices)}张裁剪至{target_train_size}张")
        else:
            print(f"[警告] 训练集原始数量{len(train_images)} < {target_train_size}，不裁剪")
 
    # 测试集裁剪（最多10000张，不足则全部保留）
        target_test_size = 10000
        if len(test_images) > target_test_size:
            np.random.seed(42)  # 固定随机种子
            test_indices = np.random.choice(len(test_images), size=target_test_size, replace=False)
            test_images = test_images[test_indices]
            test_labels = test_labels[test_indices]
            print(f"[裁剪] 测试集从{len(test_images)+len(test_indices)}张裁剪至{target_test_size}张")
        else:
            print(f"[警告] 测试集原始数量{len(test_images)} < {target_test_size}，不裁剪")
    # ==============================================



        # 数据预处理
        channel = info["n_channels"]
        # ---------------------- 修改4：图像尺寸固定为(target_size, target_size) ----------------------
        im_size = (target_size, target_size)  # 覆盖原尺寸定义
        num_classes = len(info["label"])
        class_names = [info["label"][str(i)] for i in range(num_classes)]  # 按索引0~num_classes-1生

        # ==============================================
        # 优化2：使用迭代器计算均值/标准差（无需加载全部数据到内存）
        # ==============================================
        print(f"[MedMNIST] 优化：使用迭代器计算均值/标准差（避免内存峰值）")
        # 仅使用训练集的前10000张计算统计量（平衡精度与效率）
        sample_size = min(10000, len(train_images))
        np.random.seed(42)
        sample_indices = np.random.choice(len(train_images), sample_size, replace=False)
        sample_images = train_images[sample_indices]
 
        # 动态转换样本为3通道并计算均值/标准差
        if channel == 1:
        # 单通道样本转为3通道（临时数组，计算后释放）
            sample_images_3ch = np.repeat(sample_images[..., np.newaxis], 3, axis=-1).astype(np.float32) / 255.0
        else:
            sample_images_3ch = sample_images.astype(np.float32) / 255.0
        mean = sample_images_3ch.mean(axis=(0, 1, 2)).tolist()  # (3,)
        std = sample_images_3ch.std(axis=(0, 1, 2)).tolist()    # (3,)
        del sample_images, sample_images_3ch  # 释放临时内存
 
        # 验证均值/标准差维度（保持不变）
        assert len(mean) == 3 and len(std) == 3, "均值/标准差维度错误"
        print(f"[MedMNIST] 数据集信息: 通道数={channel}, 类别数={num_classes}, 图像尺寸={im_size}")
        print(f"[MedMNIST] 优化：均值={mean}, 标准差={std}（基于样本计算）")
        # ==============================================
        # 优化3：动态数据转换（在Dataset中实时处理）
        # ============================================== 

        # ---------------------- 修改5：强制Resize至32×32 ----------------------
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(im_size),  # 无论原始尺寸，强制Resize到32×32
            transforms.ToTensor(),
            # 通道数适配：单通道→3通道，多通道保持不变
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if channel == 1 else x),
            transforms.Normalize(mean=mean, std=std)
        ])
        print(f"[MedMNIST] 调试：预处理管道={transform}")
 
        class ManualMedMNISTDataset(Dataset):
            def __init__(self, images, labels, transform):
                self.images = images  # 仅保留原始单通道数组
	        # 【修改1】保留labels为NumPy数组（不转为Tensor）
                self.labels = labels  # 直接使用NumPy数组（shape: [N,]）
                self.transform = transform
                self.targets = self.labels  # 同步为NumPy数组（兼容接口）
 
            def __len__(self):
                return len(self.images)
 
            def __getitem__(self, idx):
                img = self.images[idx]  # 原始图像（NumPy数组）
                if img.ndim == 2:  # 单通道图像添加通道维度
                    img = img[..., np.newaxis]
                img = self.transform(img)  # 图像预处理（转为Tensor）
        
        # 【核心修改2】直接访问Tensor数组的元素（已为Tensor类型）
                label = self.labels[idx].item()  # NumPy标量 → Python int
        
                # 移除调试打印以减少输出
                # if idx == 0:
                #     print(f"[MedMNIST] 优化：首个样本形状={img.shape}（期望：[3, {target_size}, {target_size}]）")
                #     print(f"[Debug] 标签类型={type(label)}, 数据类型={label}")
                return img, label
 
        print(f"[MedMNIST] 加载训练集...")
        dst_train = ManualMedMNISTDataset(train_images, train_labels, transform)
        print(f"[MedMNIST] 加载测试集...")
        dst_test = ManualMedMNISTDataset(test_images, test_labels, transform)
        print(f"[MedMNIST] 加载完成，训练集样本数: {len(dst_train)}, 测试集样本数: {len(dst_test)}")
 
    else:
        raise ValueError(f"未知数据集: {dataset}")
 
    # dataset_info定义（更新为32尺寸）
    # 确保所有数值都是Python原生类型，避免numpy或torch类型
    dataset_info = {
    'channel': int(channel),  # 确保是Python int
    'im_size': (int(im_size[0]), int(im_size[1])),  # 确保元组中的值是Python int
    'num_classes': int(num_classes),
    'classes_names': class_names,
    'mean': torch.tensor(mean),
    'std': torch.tensor(std),
    }
    # MedMNIST分支中单独添加'label'
    if dataset.startswith("MedMNIST"):
        dataset_info['label'] = info["label"]
    print(f"[MedMNIST] 调试：dataset_info包含'label'键={ 'label' in dataset_info }")
    
    # ==============================================
    # 超参数实验：验证集分割
    # ==============================================
    if hyperparameter_experiment:
        print(f"\n[超参数实验] 开始验证集分割...")
        print(f"[超参数实验] 原始训练集大小: {len(dst_train)}")
        print(f"[超参数实验] 原始测试集大小: {len(dst_test)}")
        
        # 检查训练集是否足够大
        if len(dst_train) < 10000:
            print(f"[警告] 训练集大小 ({len(dst_train)}) 小于10000，无法分出10000张作为验证集")
            print(f"[超参数实验] 将使用全部训练集作为验证集")
            validation_size = len(dst_train)
        else:
            validation_size = 10000
        
        # 获取训练集的所有索引
        train_indices = list(range(len(dst_train)))
        
        # 随机打乱索引（使用固定种子确保可复现）
        np.random.seed(42)
        np.random.shuffle(train_indices)
        
        # 分割索引：前validation_size个作为验证集，剩余作为新的训练集
        validation_indices = train_indices[:validation_size]
        new_train_indices = train_indices[validation_size:]
        
        print(f"[超参数实验] 验证集大小: {len(validation_indices)}")
        print(f"[超参数实验] 新训练集大小: {len(new_train_indices)}")
        
        # 创建验证集数据集
        if hasattr(dst_train, 'targets'):
            # 对于标准数据集
            validation_data = [dst_train[i] for i in validation_indices]
            validation_images = torch.stack([data[0] for data in validation_data])
            validation_labels = torch.tensor([data[1] for data in validation_data])
            dst_validation = TensorDataset(validation_images, validation_labels)
        else:
            # 对于自定义数据集，使用Subset
            from torch.utils.data import Subset
            dst_validation = Subset(dst_train, validation_indices)
        
        # 创建新的训练集数据集
        if hasattr(dst_train, 'targets'):
            # 对于标准数据集
            new_train_data = [dst_train[i] for i in new_train_indices]
            new_train_images = torch.stack([data[0] for data in new_train_data])
            new_train_labels = torch.tensor([data[1] for data in new_train_data])
            dst_train_new = TensorDataset(new_train_images, new_train_labels)
        else:
            # 对于自定义数据集，使用Subset
            from torch.utils.data import Subset
            dst_train_new = Subset(dst_train, new_train_indices)
        
        # 更新数据集
        dst_train = dst_train_new
        dst_test = dst_validation
        
        print(f"[超参数实验] 验证集分割完成")
        print(f"[超参数实验] 新训练集大小: {len(dst_train)}")
        print(f"[超参数实验] 新验证集（作为测试集）大小: {len(dst_test)}")
    
    #print(f"\n===== 数据集 {dataset} 加载完成（尺寸已调整为{target_size}×{target_size}） =====")
    return dataset_info, dst_train, dst_test
 

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

def find_most_similar_image(synthetic_tensor, image_library):
    """
    在图片库中找到与第一张合成图片最相近的图片张量（基于 L2 距离）。

    参数：
    - synthetic_tensor (torch.Tensor): 合成图片的张量，形状为 (batch_size, channels, height, width)。
    - image_library (torch.Tensor): 图片库的张量，形状为 (library_size, channels, height, width)。

    返回：
    - most_similar_image (torch.Tensor): 与第一张合成图片最相近的图片张量。
    - min_distance (float): 最小的 L2 距离。
    """
    # Step 1: 提取第一张合成图片的张量
    first_synthetic_image = synthetic_tensor[0].unsqueeze(0)  # 形状变为 (1, channels, height, width)

    # Step 2: 计算 L2 距离
    # 使用广播机制计算每张图片与第一张合成图片的 L2 距离
    l2_distances = torch.norm(image_library - first_synthetic_image, p=2, dim=(1, 2, 3))

    # Step 3: 找到最小距离对应的索引
    min_distance_index = torch.argmin(l2_distances)

    # Step 4: 返回最相近的图片张量
    most_similar_image = image_library[min_distance_index].unsqueeze(0)  # 形状变为 (1, channels, height, width)
    min_distance = l2_distances[min_distance_index].item()  # 获取最小距离值，并转换为 Python 标量

    return most_similar_image, min_distance

from torch.utils.data import DataLoader

class ImageDataset(Dataset):
    """自定义图像数据集类，用于加载COVIDx数据集"""
    def __init__(self, dataframe, image_folder, transform=None):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.image_folder, row['file_name'])
        image = Image.open(img_path).convert('RGB')
        label = row['class']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def sample_data_indices_from_dataset_with_restricted_indices(dst_train, data_class: str, data_num: int, restricted_indices: list = None):
    """
    从指定数据集中，基于限制的索引范围采样若干个相同类别的图片索引。

    参数：
    - dataset (Dataset): 数据集对象。
    - data_class (str): 数据类别（必须指定）。
    - data_num (int): 需要获取的图片数量。
    - restricted_indices (list, optional): 可选的索引列表，限定采样的范围。默认为 None，表示不限制索引范围。

    返回：
   
    - list: 图片索引列表。
    """
    print("sample_data_indices_from_dataset_with_restricted_indices():要获取的类为:",data_class)
    if restricted_indices is None:
        restricted_indices = list(range(len(dst_train)))  # 如果没有限制索引，则默认使用整个数据集的索引
    class_indices = []
    
    # 使用 DataLoader 来加载数据，但不打乱顺序
    dataloader = DataLoader(dst_train, batch_size=1, shuffle=False, num_workers=0)
    
    for idx, (data, label) in enumerate(dataloader):
        original_idx = idx  # 在这种情况下，idx 是全局索引
        if original_idx in restricted_indices and (label.item() == data_class or str(label.item()) == data_class):
            class_indices.append(original_idx)
            
            if len(class_indices) >= data_num:
                break

    if len(class_indices) == 0:
        print(f"错误：数据集中不存在类别 {data_class} 的数据。")
        return []

    if len(class_indices) < data_num:
        print(f"警告：数据集中类别 {data_class} 的数据数量不足 {data_num}，仅返回 {len(class_indices)} 张图片。")
        data_num = len(class_indices)

    # 使用切片获取前 data_num 个索引
    sampled_indices = class_indices[:data_num]
    print("sample_data_indices_from_dataset_with_restricted_indices():选取的sampled_indices为:", sampled_indices)
    return sampled_indices
