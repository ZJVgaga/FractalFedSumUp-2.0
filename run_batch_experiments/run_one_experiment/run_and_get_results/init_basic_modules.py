from .dataset_tools import get_dataset,setup_seed,get_dataset_indices,partition_dataset_for_clients,test_indices
from .network_tools import get_network
import torch
import torch.nn as nn
import os

def init_basic_modules(config,logger):
    """
    Initialize basic modules
    Parameters:
    - config: Configuration dictionary
    Returns:
    - dict: Dictionary containing initialized basic modules
    """
    fed_strategy = config.get("Federated_Learning_Config")
    import inspect
    current_module = inspect.getmodule(init_basic_modules)
    module_file_path = current_module.__file__
    module_file_path = os.path.dirname(module_file_path)
    if fed_strategy == "FedDM":
        logger.info("Using FedDM-specific modules")
        
        from .server.server_feddm import Server
        from .client.client_feddm import Client
        logger.info("Starting to initialize basic modules...")
    # Set random seed
        setup_seed(config.get("seed"))

    # Get dataset
        # Check if it's a hyperparameter experiment
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"Hyperparameter experiment mode: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"Loaded dataset: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[Hyperparameter Experiment] Training set size: {len(dst_train)}, Validation set (as test set) size: {len(dst_test)}")
    
    # Get dataset indices
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)
        

        

    
    # Partition client data
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"Client data partitioning completed, total {len(client_indices)} clients")
    
    # Set hardware device
        device = torch.device(config.get("device"))
        logger.info(f"Using device: {device}")
    
    # Initialize global model - using ConvNet
        logger.info("Starting to initialize FedDM classifier model (using ConvNet)...")
        from .network_tools import get_network
        
        # Initialize classifier model
        global_model = get_network("ConvNet", dataset_info, config.get("seed"), config)
        logger.info("FedDM classifier model initialization completed (using ConvNet)")
    
    # If FedDM strategy, set synthesis ratio
        def set_synth_ratio(client_classes,dst_train,config):
            """
    Calculate the total number of classes per client, total samples, and normalized total samples.
    Parameters:
    - client_classes (list of list): List of class lists for each client.
    - images_per_class (int): Number of samples per class.
    Returns:
    - client_class_counts (list): Total number of classes per client.
    - total_samples (int): Total number of samples across all clients.
    - normalized_total_samples (float): Normalized total samples (divided by 50000).
            """
            images_per_class=int(config.get("images_per_class"))
    # Step 1: Create array client_class_counts, each element represents the total number of classes for client[i]
            client_class_counts = [len(classes) for classes in client_classes]
    # Step 2: Calculate total_samples = sum(client_class_counts[i] * images_per_class)
            total_samples = sum([client_class_counts[i] * images_per_class for i in range(len(client_classes))])
    # Step 3: Calculate normalized_total_samples = total_samples / 50000
            normalized_total_samples = total_samples / len(dst_train)
            config.set("dm_synth_ratio",normalized_total_samples)
            return 
        set_synth_ratio(client_classes, dst_train,config)
        logger.info(f"FedDM synthesis ratio set to: {config.get('dm_synth_ratio')}")
    
    # Return initialized basic modules
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
    
        logger.info("Basic modules initialization completed")
        return basic_modules, Server, Client
    elif fed_strategy == "FedAvg":
        logger.info("Using FedAvg-specific modules")
        
        from .server.server_fedavg import Server
        from .client.client_fedavg import Client
        logger.info("Starting to initialize basic modules...")
    # Set random seed
        setup_seed(config.get("seed"))

    # Get dataset
        # Check if it's a hyperparameter experiment
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"Hyperparameter experiment mode: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"Loaded dataset: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[Hyperparameter Experiment] Training set size: {len(dst_train)}, Validation set (as test set) size: {len(dst_test)}")
    
    # Get dataset indices
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)
        


    # Partition client data
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"Client data partitioning completed, total {len(client_indices)} clients")
    
    # Set hardware device
        device = torch.device(config.get("device"))
        logger.info(f"Using device: {device}")
    
    # Initialize global model - using ConvNet
        logger.info("Starting to initialize FedAvg classifier model (using ConvNet)...")
        from .network_tools import get_network
        
        # Initialize classifier model
        global_model = get_network("ConvNet", dataset_info, config.get("seed"), config)
        logger.info("FedAvg classifier model initialization completed (using ConvNet)")
    
       
    # Return initialized basic modules
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
    
        logger.info("Basic modules initialization completed")
        return basic_modules, Server, Client
    elif fed_strategy == "FedAdam":
        logger.info("Using FedAdam-specific modules")
        
        from .server.server_fedadam import Server
        from .client.client_fedadam import Client
        logger.info("Starting to initialize basic modules...")
    # Set random seed
        setup_seed(config.get("seed"))

    # Get dataset
        # Check if it's a hyperparameter experiment
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"Hyperparameter experiment mode: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"Loaded dataset: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[Hyperparameter Experiment] Training set size: {len(dst_train)}, Validation set (as test set) size: {len(dst_test)}")
    
    # Get dataset indices
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)

        

    # Partition client data
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"Client data partitioning completed, total {len(client_indices)} clients")
    
    # Set hardware device
        device = torch.device(config.get("device"))
        logger.info(f"Using device: {device}")
    
    # Initialize global model - using ConvNet
        logger.info("Starting to initialize FedAdam classifier model (using ConvNet)...")
        from .network_tools import get_network
        
        # Initialize classifier model
        global_model = get_network("ConvNet", dataset_info, config.get("seed"), config)
        logger.info("FedAdam classifier model initialization completed (using ConvNet)")
    
       
    # Return initialized basic modules
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
    
        logger.info("Basic modules initialization completed")
        return basic_modules, Server, Client
    elif fed_strategy == "FedProx":
        logger.info("Using FedProx-specific modules")
        
        from .server.server_fedprox import Server
        from .client.client_fedprox import Client
        logger.info("Starting to initialize basic modules...")
    # Set random seed
        setup_seed(config.get("seed"))

    # Get dataset
        # Check if it's a hyperparameter experiment
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"Hyperparameter experiment mode: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"Loaded dataset: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[Hyperparameter Experiment] Training set size: {len(dst_train)}, Validation set (as test set) size: {len(dst_test)}")
    
    # Get dataset indices
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)
        

    
    # Partition client data
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"Client data partitioning completed, total {len(client_indices)} clients")
    
    # Set hardware device
        device = torch.device(config.get("device"))
        logger.info(f"Using device: {device}")
    
    # Initialize global model - using ConvNet
        logger.info("Starting to initialize FedProx classifier model (using ConvNet)...")
        from .network_tools import get_network
        
        # Initialize classifier model
        global_model = get_network("ConvNet", dataset_info, config.get("seed"), config)
        logger.info("FedProx classifier model initialization completed (using ConvNet)")
    
       
    # Return initialized basic modules
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
    
        logger.info("Basic modules initialization completed")
        return basic_modules, Server, Client
    elif fed_strategy == "FedSumUp":
        logger.info("Using FedSumUp-specific modules")
        
        from .server.server_fedsumup import Server
        from .client.client_fedsumup import Client
        logger.info("Starting to initialize basic modules...")
    # Set random seed
        setup_seed(config.get("seed"))

    # Get dataset
        # Check if it's a hyperparameter experiment
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"Hyperparameter experiment mode: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"Loaded dataset: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[Hyperparameter Experiment] Training set size: {len(dst_train)}, Validation set (as test set) size: {len(dst_test)}")
    
    # Get dataset indices
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)
        
        # Get target_train_labels
        target_train_labels = [dst_train[i][1] for i in target_train_indices]
        
        logger.info("Dataset index partitioning completed")
        logger.info(f"target_train_indices count: {len(target_train_indices)}, target_train_labels count: {len(target_train_labels)}")
    
    # Partition client data
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"Client data partitioning completed, total {len(client_indices)} clients")
    
    # Set hardware device
        device = torch.device(config.get("device"))
        logger.info(f"Using device: {device}")
    

    
    # Initialize funnel encoder (FedSumUp requires) - using new funnel encoder
        logger.info("Starting to initialize funnel encoder (new process)...")
        from .funnel_encoder import get_funnel_encoder_for_fedsumup
        
        # Get funnel encoder
        vae = get_funnel_encoder_for_fedsumup(dataset_info, config)
        vae.to(device)
        vae.eval()
        logger.info("Funnel encoder initialization completed")
        
        # Initialize classifier model (FedSumUp requires) - using ConvNet
        logger.info("Starting to initialize FedSumUp classifier model (using ConvNet)...")
        from .network_tools import get_network
        
        # Initialize ConvNet classifier
        classifier = get_network("ConvNet", dataset_info, config.get("seed"), config)
        classifier.to(device)
        logger.info("FedSumUp classifier model initialization completed (using ConvNet)")
    
    # Return initialized basic modules
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
      
        'vae': vae,  # Add VAE to basic modules
        'classifier': classifier  # Add classifier to basic modules
        }
    
        logger.info("Basic modules initialization completed")
        return basic_modules, Server, Client
    elif fed_strategy == "FedSum":
        logger.info("Using FedSum-specific modules (client_fedsum/server_fedsum)")
        
        from .server.server_fedsum import Server
        from .client.client_fedsum import Client
        logger.info("Starting to initialize basic modules...")
    # Set random seed
        setup_seed(config.get("seed"))

    # Get dataset
        # Check if it's a hyperparameter experiment
        hyperparameter_experiment = config.get("hyperparameter_experiment", False)
        logger.info(f"Hyperparameter experiment mode: {hyperparameter_experiment}")
        
        dataset_info, dst_train, dst_test = get_dataset(
            config.get("Dataset"), 
            module_file_path+"/data",
            hyperparameter_experiment=hyperparameter_experiment
        )
        logger.info(f"Loaded dataset: {config.get('Dataset')}")
        if hyperparameter_experiment:
            logger.info(f"[Hyperparameter Experiment] Training set size: {len(dst_train)}, Validation set (as test set) size: {len(dst_test)}")
    
    # Get dataset indices
        target_train_indices, shadow_train_indices, target_test_indices, shadow_test_indices = get_dataset_indices(dst_train, dst_test,config)
        

    
    # Partition client data
        client_indices, client_classes = partition_dataset_for_clients(dataset_info, dst_train, target_train_indices, config.get("alpha"),config)
        logger.info(f"Client data partitioning completed, total {len(client_indices)} clients")
    
    # Set hardware device
        device = torch.device(config.get("device"))
        logger.info(f"Using device: {device}")
    
    # Initialize global model - using ConvNet (consistent with FedAvg)
        logger.info