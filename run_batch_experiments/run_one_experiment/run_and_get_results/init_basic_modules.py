from .dataset_tools import get_dataset,setup_seed,get_dataset_indices,partition_dataset_for_clients,test_indices
from .network_tools import get_network
import torch
import torch.nn as nn
import os

def init_basic_modules(config,logger):
    """
    初始化基本模块
    参数：
    - config: 配置字典
    返回：
    - dict: 包含初始化好的基本模块的字典
    """
    fed_strategy = config.get("Federated_Learning_Config")
    import inspect
    current_module = inspect.getmodule(init_basic_modules)
    module_file_path = current_module.__file__
    module_file_path = os.path.dirname(module_file_path)
    if fed_strategy == "FedDM":
        logger.info("使用FedDM专用模块")
        
        from .server.server_feddm import Server
        from .client.client_feddm import Client
        logger.info("开始初始化基本模块...")
    # 设置随机种子
        setup_seed(config.get("seed"))

    # 获取数据集
        # 检查是否为超参数实验
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"超参数实验模式: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"已加载数据集: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[超参数实验] 训练集大小: {len(dst_train)}, 验证集（作为测试集）大小: {len(dst_test)}")
    
    # 获取数据集索引
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)
        

        

    
    # 划分客户端数据
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"已完成客户端数据划分，共{len(client_indices)}个客户端")
    
    # 设置硬件设备
        device = torch.device(config.get("device"))
        logger.info(f"使用设备: {device}")
    
    # 初始化全局模型 - 使用ConvNet
        logger.info("开始初始化FedDM分类器模型（使用ConvNet）...")
        from .network_tools import get_network
        
        # 初始化分类器模型
        global_model = get_network("ConvNet", dataset_info, config.get("seed"), config)
        logger.info("FedDM分类器模型初始化完成（使用ConvNet）")
    
    # 如果是FedDM策略，设置合成比例
        def set_synth_ratio(client_classes,dst_train,config):
            """
    计算每个客户端的类别总数、总样本数和归一化后的总样本数。
    参数：
    - client_classes (list of list): 每个客户端的类别列表。
    - images_per_class (int): 每类样本的数量。
    返回：
    - client_class_counts (list): 每个客户端的类别总数。
    - total_samples (int): 所有客户端的总样本数。
    - normalized_total_samples (float): 归一化后的总样本数（除以50000）。
            """
            images_per_class=int(config.get("images_per_class"))
    # Step 1: 创建数组 client_class_counts，每个元素代表 client[i] 的总类数
            client_class_counts = [len(classes) for classes in client_classes]
    # Step 2: 计算 total_samples = sum(client_class_counts[i] * images_per_class)
            total_samples = sum([client_class_counts[i] * images_per_class for i in range(len(client_classes))])
    # Step 3: 计算 normalized_total_samples = total_samples / 50000
            normalized_total_samples = total_samples / len(dst_train)
            config.set("dm_synth_ratio",normalized_total_samples)
            return 
        set_synth_ratio(client_classes, dst_train,config)
        logger.info(f"已设置FedDM合成比例为: {config.get('dm_synth_ratio')}")
    
    # 返回初始化好的基本模块
        basic_modules = {
        'dataset_info': dataset_info,
        'dst_train': dst_train,
        'dst_test': dst_test,
        'target_train_indices': target_train_indices,
        'shadow_train_indices': shadow_train_indices,
        'target_test_indices': target_test_indices,
        'shadow_test_indices': shadow_test_indices,
        'client_indices': client_indices,
        'client_classes': client_classes,
        'communication_rounds':config.get('communication_rounds'),
        'device': device,
        'global_model': global_model
        }
    
        logger.info("基本模块初始化完成")
        return basic_modules, Server, Client
    elif fed_strategy == "FedAvg":
        logger.info("使用FedAvg专用模块")
        
        from .server.server_fedavg import Server
        from .client.client_fedavg import Client
        logger.info("开始初始化基本模块...")
    # 设置随机种子
        setup_seed(config.get("seed"))

    # 获取数据集
        # 检查是否为超参数实验
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"超参数实验模式: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"已加载数据集: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[超参数实验] 训练集大小: {len(dst_train)}, 验证集（作为测试集）大小: {len(dst_test)}")
    
    # 获取数据集索引
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)
        


    # 划分客户端数据
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"已完成客户端数据划分，共{len(client_indices)}个客户端")
    
    # 设置硬件设备
        device = torch.device(config.get("device"))
        logger.info(f"使用设备: {device}")
    
    # 初始化全局模型 - 使用ConvNet
        logger.info("开始初始化FedAvg分类器模型（使用ConvNet）...")
        from .network_tools import get_network
        
        # 初始化分类器模型
        global_model = get_network("ConvNet", dataset_info, config.get("seed"), config)
        logger.info("FedAvg分类器模型初始化完成（使用ConvNet）")
    
       
    # 返回初始化好的基本模块
        basic_modules = {
        'dataset_info': dataset_info,
        'dst_train': dst_train,
        'dst_test': dst_test,
        'target_train_indices': target_train_indices,
        'shadow_train_indices': shadow_train_indices,
        'target_test_indices': target_test_indices,
        'shadow_test_indices': shadow_test_indices,
        'client_indices': client_indices,
        'client_classes': client_classes,
        'communication_rounds':config.get('communication_rounds'),
        'device': device,
        'global_model': global_model
        }
    
        logger.info("基本模块初始化完成")
        return basic_modules, Server, Client
    elif fed_strategy == "FedAdam":
        logger.info("使用FedAdam专用模块")
        
        from .server.server_fedadam import Server
        from .client.client_fedadam import Client
        logger.info("开始初始化基本模块...")
    # 设置随机种子
        setup_seed(config.get("seed"))

    # 获取数据集
        # 检查是否为超参数实验
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"超参数实验模式: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"已加载数据集: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[超参数实验] 训练集大小: {len(dst_train)}, 验证集（作为测试集）大小: {len(dst_test)}")
    
    # 获取数据集索引
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)

        

    # 划分客户端数据
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"已完成客户端数据划分，共{len(client_indices)}个客户端")
    
    # 设置硬件设备
        device = torch.device(config.get("device"))
        logger.info(f"使用设备: {device}")
    
    # 初始化全局模型 - 使用ConvNet
        logger.info("开始初始化FedAdam分类器模型（使用ConvNet）...")
        from .network_tools import get_network
        
        # 初始化分类器模型
        global_model = get_network("ConvNet", dataset_info, config.get("seed"), config)
        logger.info("FedAdam分类器模型初始化完成（使用ConvNet）")
    
       
    # 返回初始化好的基本模块
        basic_modules = {
        'dataset_info': dataset_info,
        'dst_train': dst_train,
        'dst_test': dst_test,
        'target_train_indices': target_train_indices,
        'shadow_train_indices': shadow_train_indices,
        'target_test_indices': target_test_indices,
        'shadow_test_indices': shadow_test_indices,
        'client_indices': client_indices,
        'client_classes': client_classes,
        'communication_rounds':config.get('communication_rounds'),
        'device': device,
        'global_model': global_model
        }
    
        logger.info("基本模块初始化完成")
        return basic_modules, Server, Client
    elif fed_strategy == "FedProx":
        logger.info("使用FedProx专用模块")
        
        from .server.server_fedprox import Server
        from .client.client_fedprox import Client
        logger.info("开始初始化基本模块...")
    # 设置随机种子
        setup_seed(config.get("seed"))

    # 获取数据集
        # 检查是否为超参数实验
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"超参数实验模式: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"已加载数据集: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[超参数实验] 训练集大小: {len(dst_train)}, 验证集（作为测试集）大小: {len(dst_test)}")
    
    # 获取数据集索引
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)
        

    
    # 划分客户端数据
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"已完成客户端数据划分，共{len(client_indices)}个客户端")
    
    # 设置硬件设备
        device = torch.device(config.get("device"))
        logger.info(f"使用设备: {device}")
    
    # 初始化全局模型 - 使用ConvNet
        logger.info("开始初始化FedProx分类器模型（使用ConvNet）...")
        from .network_tools import get_network
        
        # 初始化分类器模型
        global_model = get_network("ConvNet", dataset_info, config.get("seed"), config)
        logger.info("FedProx分类器模型初始化完成（使用ConvNet）")
    
       
    # 返回初始化好的基本模块
        basic_modules = {
        'dataset_info': dataset_info,
        'dst_train': dst_train,
        'dst_test': dst_test,
        'target_train_indices': target_train_indices,
        'shadow_train_indices': shadow_train_indices,
        'target_test_indices': target_test_indices,
        'shadow_test_indices': shadow_test_indices,
        'client_indices': client_indices,
        'client_classes': client_classes,
        'communication_rounds':config.get('communication_rounds'),
        'device': device,
        'global_model': global_model
        }
    
        logger.info("基本模块初始化完成")
        return basic_modules, Server, Client
    elif fed_strategy == "FedSumUp":
        logger.info("使用FedSumUp专用模块")
        
        from .server.server_fedsumup import Server
        from .client.client_fedsumup import Client
        logger.info("开始初始化基本模块...")
    # 设置随机种子
        setup_seed(config.get("seed"))

    # 获取数据集
        # 检查是否为超参数实验
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"超参数实验模式: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"已加载数据集: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[超参数实验] 训练集大小: {len(dst_train)}, 验证集（作为测试集）大小: {len(dst_test)}")
    
    # 获取数据集索引
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)
        
        # 获取target_train_labels
        target_train_labels = [dst_train[i][1] for i in target_train_indices]
        
        logger.info("已完成数据集索引划分")
        logger.info(f"target_train_indices数量: {len(target_train_indices)}, target_train_labels数量: {len(target_train_labels)}")
    
    # 划分客户端数据
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"已完成客户端数据划分，共{len(client_indices)}个客户端")
    
    # 设置硬件设备
        device = torch.device(config.get("device"))
        logger.info(f"使用设备: {device}")
    

    
    # 初始化漏斗编码器（FedSumUp需要）- 使用新的漏斗编码器
        logger.info("开始初始化漏斗编码器（新流程）...")
        from .funnel_encoder import get_funnel_encoder_for_fedsumup
        
        # 获取漏斗编码器
        vae = get_funnel_encoder_for_fedsumup(dataset_info, config)
        vae.to(device)
        vae.eval()
        logger.info("漏斗编码器初始化完成")
        
        # 初始化分类器模型（FedSumUp需要）- 使用ConvNet
        logger.info("开始初始化FedSumUp分类器模型（使用ConvNet）...")
        from .network_tools import get_network
        
        # 初始化ConvNet分类器
        classifier = get_network("ConvNet", dataset_info, config.get("seed"), config)
        classifier.to(device)
        logger.info("FedSumUp分类器模型初始化完成（使用ConvNet）")
    
    # 返回初始化好的基本模块
        basic_modules = {
        'dataset_info': dataset_info,
        'dst_train': dst_train,
        'dst_test': dst_test,
        'target_train_indices': target_train_indices,
        'target_train_labels': target_train_labels,
        'shadow_train_indices': shadow_train_indices,
        'target_test_indices': target_test_indices,
        'shadow_test_indices': shadow_test_indices,
        'client_indices': client_indices,
        'client_classes': client_classes,
        'communication_rounds':config.get("communication_rounds"),
        'device': device,
      
        'vae': vae,  # 添加VAE到基本模块
        'classifier': classifier  # 添加分类器到基本模块
        }
    
        logger.info("基本模块初始化完成")
        return basic_modules, Server, Client
    elif fed_strategy == "FedSum":
        logger.info("使用FedSum专用模块（client_fedsum/server_fedsum）")
        
        from .server.server_fedsum import Server
        from .client.client_fedsum import Client
        logger.info("开始初始化基本模块...")
    # 设置随机种子
        setup_seed(config.get("seed"))

    # 获取数据集
        # 检查是否为超参数实验
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"超参数实验模式: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"已加载数据集: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[超参数实验] 训练集大小: {len(dst_train)}, 验证集（作为测试集）大小: {len(dst_test)}")
    
    # 获取数据集索引
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)
        

    
    # 划分客户端数据
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"已完成客户端数据划分，共{len(client_indices)}个客户端")
    
    # 设置硬件设备
        device = torch.device(config.get("device"))
        logger.info(f"使用设备: {device}")
    
    # 初始化全局模型 - 使用ConvNet（和FedAvg保持一致）
        logger.info("开始初始化FedSum分类器模型（使用ConvNet，和FedAvg保持一致）...")
        from .network_tools import get_network
        
        # 初始化分类器模型
        global_model = get_network("ConvNet", dataset_info, config.get("seed"), config)
        logger.info("FedSum分类器模型初始化完成（使用ConvNet）")
    
        # 初始化VAE模型（和FedSD2C保持一致）- 直接从本地加载
        logger.info("开始初始化VAE模型（和FedSD2C保持一致）- 直接从本地加载...")
        from diffusers import AutoencoderKL
        
        # 尝试多个可能的本地路径
        local_paths = [
            os.path.join(module_file_path, "model", "sdxl-vae"),
            "FractalFedSumUp/run_batch_experiments/run_one_experiment/run_and_get_results/model/sdxl-vae",
            os.path.expanduser("~/.cache/huggingface/hub/models--stabilityai--sdxl-vae"),
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--stabilityai--sdxl-vae")
        ]
        
        vae_loaded = False
        for local_path in local_paths:
            logger.info(f"尝试从本地路径加载VAE: {local_path}")
            if os.path.exists(local_path):
                try:
                    vae = AutoencoderKL.from_pretrained(
                        local_path,
                        use_safetensors=True,
                        local_files_only=True
                    )
                    vae.to(device)
                    vae.eval()
                    logger.info(f"VAE从本地路径加载成功: {local_path}")
                    vae_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"从本地路径 {local_path} 加载VAE失败: {e}")
        
        if not vae_loaded:
            raise FileNotFoundError(f"无法从任何本地路径加载VAE模型。请确保VAE模型已下载到以下路径之一: {local_paths}")
    
    # 返回初始化好的基本模块
        basic_modules = {
        'dataset_info': dataset_info,
        'dst_train': dst_train,
        'dst_test': dst_test,
        'target_train_indices': target_train_indices,
        'shadow_train_indices': shadow_train_indices,
        'target_test_indices': target_test_indices,
        'shadow_test_indices': shadow_test_indices,
        'client_indices': client_indices,
        'client_classes': client_classes,
        'communication_rounds':config.get('communication_rounds'),
        'device': device,
        'global_model': global_model,
        'vae': vae  # 添加VAE到基本模块
        }
    
        logger.info("基本模块初始化完成")
        return basic_modules, Server, Client
    
    elif fed_strategy == "FedSD2C":
        logger.info("使用FedSD2C专用模块")
        
        from .server.server_fedsd2c import Server
        from .client.client_fedsd2c import Client
        logger.info("开始初始化基本模块...")
    # 设置随机种子
        setup_seed(config.get("seed"))

    # 获取数据集
        # 检查是否为超参数实验
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"超参数实验模式: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"已加载数据集: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[超参数实验] 训练集大小: {len(dst_train)}, 验证集（作为测试集）大小: {len(dst_test)}")
    
    # 获取数据集索引
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)
        logger.info("已完成数据集索引划分")
    
    # 划分客户端数据
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"已完成客户端数据划分，共{len(client_indices)}个客户端")
    
    # 设置硬件设备
        device = torch.device(config.get("device"))
        logger.info(f"使用设备: {device}")
    
    # 初始化全局模型 - 使用ConvNet
        logger.info("开始初始化FedSD2C分类器模型（使用ConvNet）...")
        from .network_tools import get_network
        
        # 初始化分类器模型
        global_model = get_network("ConvNet", dataset_info, config.get("seed"), config)
        logger.info("FedSD2C分类器模型初始化完成（使用ConvNet）")
    
        # 初始化VAE模型（FedSD2C需要）
        logger.info("开始初始化VAE模型...")
        from diffusers import AutoencoderKL
        
        # 首先尝试从本地model文件夹加载
        local_model_path = os.path.join(module_file_path, "model", "sdxl-vae")
        logger.info(f"尝试从本地model文件夹加载VAE: {local_model_path}")
        
        try:
            if os.path.exists(local_model_path):
                vae = AutoencoderKL.from_pretrained(
                    local_model_path,
                    use_safetensors=True,
                    local_files_only=True
                )
                vae.to(device)
                vae.eval()
                logger.info("VAE从本地model文件夹加载成功")
            else:
                logger.warning(f"本地model文件夹不存在: {local_model_path}")
                # 如果本地路径不存在，使用用户提供的绝对路径
                user_local_path = "FractalFedSumUp/run_batch_experiments/run_one_experiment/run_and_get_results/model/sdxl-vae"
                logger.info(f"尝试使用用户提供的路径: {user_local_path}")
                if os.path.exists(user_local_path):
                    vae = AutoencoderKL.from_pretrained(
                        user_local_path,
                        use_safetensors=True,
                        local_files_only=True
                    )
                    vae.to(device)
                    vae.eval()
                    logger.info("VAE从用户提供的路径加载成功")
                else:
                    raise FileNotFoundError(f"VAE模型文件夹不存在: {local_model_path} 和 {user_local_path}")
                
        except Exception as e:
            logger.warning(f"从本地路径加载VAE失败: {e}")
            logger.info("尝试从Hugging Face缓存加载...")
            
            # 设置环境变量，优先使用本地缓存
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            try:
                vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sdxl-vae", 
                    use_safetensors=False,
                    local_files_only=True  # 只使用本地文件
                )
                vae.to(device)
                vae.eval()
                logger.info("VAE从Hugging Face缓存加载成功")
            except Exception as e2:
                logger.warning(f"从Hugging Face缓存加载VAE失败: {e2}")
                logger.info("尝试在线下载VAE...")
                # 如果本地缓存没有，尝试在线下载
                os.environ['HF_HUB_OFFLINE'] = '0'
                os.environ['TRANSFORMERS_OFFLINE'] = '0'
                vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sdxl-vae", 
                    use_safetensors=False,
                    local_files_only=False
                )
                vae.to(device)
                vae.eval()
                logger.info("VAE在线下载成功")
    
    # 返回初始化好的基本模块
        basic_modules = {
        'dataset_info': dataset_info,
        'dst_train': dst_train,
        'dst_test': dst_test,
        'target_train_indices': target_train_indices,
        'shadow_train_indices': shadow_train_indices,
        'target_test_indices': target_test_indices,
        'shadow_test_indices': shadow_test_indices,
        'client_indices': client_indices,
        'client_classes': client_classes,
        'communication_rounds':1,
        'device': device,
        'global_model': global_model,
        'vae': vae  # 添加VAE到基本模块
        }
    
        logger.info("基本模块初始化完成")
        return basic_modules, Server, Client
    
    elif fed_strategy == "CollabDM":
        logger.info("使用CollabDM专用模块")
        
        from .server.server_collabdm import Server
        from .client.client_collabdm import Client
        logger.info("开始初始化基本模块...")
    # 设置随机种子
        setup_seed(config.get("seed"))

    # 获取数据集
        # 检查是否为超参数实验
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"超参数实验模式: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"已加载数据集: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[超参数实验] 训练集大小: {len(dst_train)}, 验证集（作为测试集）大小: {len(dst_test)}")
    
    # 获取数据集索引
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)
        logger.info("已完成数据集索引划分")
    
    # 划分客户端数据
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"已完成客户端数据划分，共{len(client_indices)}个客户端")
    
    # 设置硬件设备
        device = torch.device(config.get("device"))
        logger.info(f"使用设备: {device}")
        from .network_tools import get_network
        # 初始化分类器模型
        global_model = get_network("ConvNet", dataset_info, config.get("seed"), config)
        logger.info("CollabDM分类器模型初始化完成（使用ConvNet）")
    
    # 如果是FedSD2C策略，设置合成比例
       
    # 返回初始化好的基本模块
        basic_modules = {
        'dataset_info': dataset_info,
        'dst_train': dst_train,
        'dst_test': dst_test,
        'target_train_indices': target_train_indices,
        'shadow_train_indices': shadow_train_indices,
        'target_test_indices': target_test_indices,
        'shadow_test_indices': shadow_test_indices,
        'client_indices': client_indices,
        'client_classes': client_classes,
        'communication_rounds':1,
        'device': device,
        'global_model': global_model
        }
    
        logger.info("基本模块初始化完成")
        return basic_modules, Server, Client
    
    
    else:
        raise ValueError(f"不支持的联邦学习策略: {fed_strategy}")
