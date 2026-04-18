import os
import time

from .attack.mia.Train_Logits_Inference_Model.MIA import train_inference_model, prepare_and_evaluate_attack, load_fc_net_list, save_fc_net_list
from .process_tools import measure_size, early_stopper
from .init_basic_modules import init_basic_modules


def initialize_experiment_components(config, logger):
    """
    Initialize the basic components required for the experiment (server and clients).
    
    Parameters:
    - config (dict): Experiment configuration dictionary, containing configurations such as client_num
    - logger: Logger instance
    
    Returns:
    - tuple: (basic_modules, server, client_list)
      - basic_modules (dict): Dictionary of basic modules
      - server (Server): Server instance
      - client_list (list): List of client instances, format: [Client1, Client2, ...]
    """
    # First, initialize the required basic modules
    basic_modules, Server, Client = init_basic_modules(config, logger)
    
    # Initialize the client list (initialize client_num clients simultaneously)
    client_list = []
    for i in range(config.get("client_num")):
        client_i = Client(basic_modules, config, logger, i)
        client_list.append(client_i)
    
    # Initialize the Server
    server = Server(basic_modules, config, logger, client_list)
    logger.info('Server and Clients have been created.')
    
    return basic_modules, server, client_list


def create_experiment_result_structure():
    """
    Create and initialize the experiment result data structure.
    
    Returns:
    - dict: Initialized experiment result dictionary, containing the following fields:
      - total_communication_size (float): Total communication size (bytes)
      - client_train_time (float): Total client training time (seconds)
      - round_accuracies (list): List of accuracy per round, format: [acc1, acc2, ...]
      - round_mia_accuracies (list): List of MIA accuracy per round, format: [mia_acc1, mia_acc2, ...]
      - client_compute_budgets (list): List of client compute budget usage per round,
        format: [[{client_stats1}, {client_stats2}, ...], ...]
    """
    return {
        "total_communication_size": 0,
        "client_train_time": 0,
        "round_accuracies": [],
        "round_mia_accuracies": [],  # New: MIA accuracy per round
        "client_compute_budgets": [],  # New: Client compute budget usage per round
    }


def load_or_train_mia_model(basic_modules, config, logger):
    """
    Load or train the MIA (Membership Inference Attack) shadow model.
    
    Parameters:
    - basic_modules (dict): Dictionary of basic modules
    - config (dict): Experiment configuration dictionary
    - logger: Logger instance
    
    Returns:
    - list or None: List of MIA shadow models, format: [fc_net1, fc_net2, ...]
    """
    logger.info('Starting to load MIA shadow model...')
    fc_net_list = load_fc_net_list(config)  # Attempt to load existing models
    
    # TODO: Add MIA detection later. If it's not an MIA attack method, there's no need to train shadow models.
    # Train shadow models
    if fc_net_list is None: 
        logger.info('MIA shadow model does not exist, starting to train shadow model...')
        fc_net_list = train_inference_model(basic_modules, config)
        logger.info('MIA shadow model training completed, starting to save...')
        save_fc_net_list(fc_net_list, config)
        logger.info('MIA shadow model saving completed')
    else:
        logger.info('MIA shadow model loaded successfully')
    
    return fc_net_list


def setup_gpu_monitoring(gpu_monitor_dict, logger):
    """
    Set up GPU monitoring related parameters.
    
    Parameters:
    - gpu_monitor_dict (dict or None): GPU monitoring communication dictionary, containing command_queue and result_dict
    - logger: Logger instance
    
    Returns:
    - tuple: (command_queue, result_dict, process_pid, client_total_sm_seconds, 
              client_total_cpu_time, client_total_wall_time)
      - command_queue (Queue or None): Command queue
      - result_dict (dict or None): Result dictionary
      - process_pid (int): Process PID
      - client_total_sm_seconds (list): Cumulative GPU SM·seconds for all clients (using list as mutable container)
      - client_total_cpu_time (list): Cumulative CPU time for all clients (using list as mutable container)
      - client_total_wall_time (list): Cumulative wall-clock time for all clients (using list as mutable container)
    """
    # Initialize GPU monitoring communication, using lists as mutable containers
    client_total_sm_seconds = [0.0]  # Cumulative GPU SM·seconds for all clients
    client_total_cpu_time = [0.0]    # Cumulative CPU time for all clients
    client_total_wall_time = [0.0]   # Cumulative wall-clock time for all clients
    
    # If GPU monitoring dictionary is provided, use it to measure GPU SM utilization
    if gpu_monitor_dict:
        logger.info(f'gpu_monitor_dict exists: {gpu_monitor_dict}')
        command_queue = gpu_monitor_dict.get('command_queue')
        result_dict = gpu_monitor_dict.get('result_dict')
        process_pid = gpu_monitor_dict.get('process_pid', os.getpid())
        
        # Check if command_queue and result_dict are valid
        if command_queue and result_dict:
            logger.info(f'Enabling GPU SM utilization monitoring, process PID: {process_pid}')
            logger.info(f'command_queue type: {type(command_queue)}')
            logger.info(f'result_dict type: {type(result_dict)}')
        else:
            command_queue = None
            result_dict = None
            process_pid = os.getpid()
            logger.info('GPU monitoring dictionary exists but command_queue or result_dict is invalid')
    else:
        command_queue = None
        result_dict = None
        process_pid = os.getpid()
        logger.info('GPU SM utilization monitoring not enabled (gpu_monitor_dict does not exist)')
    
    return command_queue, result_dict, process_pid, client_total_sm_seconds, client_total_cpu_time, client_total_wall_time


def get_actual_rounds(basic_modules, config, logger):
    """
    Get the actual number of training rounds.
    
    Parameters:
    - basic_modules (dict): Dictionary of basic modules
    - config (dict): Experiment configuration dictionary
    - logger: Logger instance
    
    Returns:
    - int: Actual number of training rounds
    """
    # For FedSumUp, the new version requires only one communication per round
    fed_strategy = config.get("Federated_Learning_Config")
    if fed_strategy == "FedSumUp":
        actual_rounds = basic_modules["communication_rounds"]
        logger.info(f"FedSumUp strategy (new version): Only one communication per round, number of rounds: {actual_rounds}")
    else:
        actual_rounds = basic_modules["communication_rounds"]
    
    return actual_rounds


def measure_gpu_utilization(command_queue, result_dict, process_pid, operation_type, logger):
    """
    Measure GPU utilization.
    
    Parameters:
    - command_queue (Queue or None): GPU monitoring command queue
    - result_dict (dict or None): GPU monitoring result dictionary
    - process_pid (int): Process PID
    - operation_type (str): Operation type, 'start' or 'end'
    - logger: Logger instance
    
    Returns:
    - float: Cumulative GPU SM utilization value (only returned when operation_type is 'end')
    """
    if not command_queue or not result_dict:
        logger.debug(f"GPU monitoring not enabled, skipping {operation_type} operation")
        return 0.0
    
    # Send command
    command_queue.put((operation_type, process_pid, os.getpid()))
    logger.info(f'Sending GPU monitoring {operation_type} command, PID: {process_pid}')
    
    # Wait for result
    start_time = time.time()
    sm_accumulated = 0.0
    
    while time.time() - start_time < 5.0:  # Wait up to 5 seconds
        result_key = f"result_{os.getpid()}_{process_pid}_{operation_type}"
        if result_key in result_dict:
            result = result_dict[result_key]
            if operation_type == 'end' and len(result) >= 3:
                sm_accumulated = result[2]  # Get cumulative value
                logger.info(f'Received GPU monitoring {operation_type} result: {sm_accumulated}')
            elif operation_type == 'start':
                logger.info(f'Received GPU monitoring {operation_type} result: {result}')
            break
        time.sleep(0.1)
    
    return sm_accumulated


def process_client_training(client, server, command_queue, result_dict, process_pid, 
                           client_total_sm_seconds, client_total_cpu_time, client_total_wall_time,
                           experiment_result, config, logger):
    """
    Process the training of a single client, including GPU monitoring.
    
    Parameters:
    - client (Client): Client instance
    - server (Server): Server instance
    - command_queue (Queue or None): GPU monitoring command queue
    - result_dict (dict or None): GPU monitoring result dictionary
    - process_pid (int): Process PID
    - client_total_sm_seconds (list): Cumulative GPU SM·seconds (using list as mutable container)
    - client_total_cpu_time (list): Cumulative CPU time (using list as mutable container)
    - client_total_wall_time (list): Cumulative wall-clock time (using list as mutable container)
    - experiment_result (dict): Experiment result dictionary (will be updated)
    - config (dict): Experiment configuration dictionary
    - logger: Logger instance
    
    Returns:
    - dict: Data returned by the client, format: {data_key: data_value, ...}
    """
    # User-customizable process (start)
    server_data = server.arrange_server_data_to_client()
    client.receive_data_from_server(server_data)
    
    # 1. Use the encapsulated GPU measurement function
    # Start GPU monitoring
    measure_gpu_utilization(command_queue, result_dict, process_pid, 'start', logger)
    
    # Record CPU time start
    cpu_start_time = time.process_time()
    # Record wall-clock time start
    wall_start_time = time.time()
    
    # 2. Execute client processing
    # Note: client.process() internally handles compute budget, no need to pass parameters
    client.process()
    
    # Record CPU time end
    cpu_end_time = time.process_time()
    # Record wall-clock time end
    wall_end_time = time.time()
    
    # Update cumulative times (update via list index)
    client_cpu_time_delta = cpu_end_time - cpu_start_time
    client_total_cpu_time[0] += client_cpu_time_delta
    
    client_wall_time_delta = wall_end_time - wall_start_time
    client_total_wall_time[0] += client_wall_time_delta
    
    # End GPU monitoring and get cumulative value
    sm_accumulated = measure_gpu_utilization(command_queue, result_dict, process_pid, 'end', logger)
    client_total_sm_seconds[0] += sm_accumulated
    
    # Log detailed CPU time information
    logger.info(f"Client {client.client_id if hasattr(client, 'client_id') else 'unknown'} CPU time: {client_cpu_time_delta:.4f} seconds, Cumulative CPU time: {client_total_cpu_time[0]:.4f} seconds")
    
    # 3. Get data returned by the client
    received_data = client.send_data_to_server()
    
    # 4. Calculate information size from client to server for evaluating communication overhead
    experiment_result["total_communication_size"] += (measure_size(server_data) + measure_size(received_data))
    
    return received_data


def process_server_round(server, received_data_list, basic_modules, config, 
                        fc_net_list, experiment_result, rounds, logger):
    """
    Process server-side aggregation and update.
    
    Parameters:
    - server (Server): Server instance
    - received_data_list (list): List of data returned by clients, format: [{data1}, {data2}, ...]
    - basic_modules (dict): Dictionary of basic modules
    - config (dict): Experiment configuration dictionary
    - fc_net_list (list): List of MIA shadow models
    - experiment_result (dict): Experiment result dictionary (will be updated)
    - rounds (int): Current round number
    - logger: Logger instance
    """
    logger.info('---------- server process ----------')
    # User-customizable process (start)
    merged_data = server.merge_data(received_data_list)
    server.process(merged_data)
    
    # Calculate and record MIA accuracy for this round
    mia_accuracy = prepare_and_evaluate_attack(basic_modules, config, server.global_model, fc_net_list)
    logger.info(f"MIA attack accuracy is: {mia_accuracy}")
    experiment_result["round_mia_accuracies"].append(mia_accuracy)  # Record MIA accuracy per round
    
    # Collect compute budget usage of clients for this round
    round_budget_stats = []
    for received_data in received_data_list:
        if "compute_budget_status" in received_data:
            budget_status = received_data["compute_budget_status"]
            round_budget_stats.append({
                "client_id": budget_status.get("client_id", -1),
                "current_flops": budget_status.get("current_flops", 0.0),
                "budget_usage_percent": budget_status.get("budget_usage_percent", 0.0),
                "budget_exceeded": budget_status.get("budget_exceeded", False),
                "epoch_count": budget_status.get("epoch_count", 0)
            })
    
    experiment_result["client_compute_budgets"].append(round_budget_stats)
    logger.info(f"Round {rounds} client compute budget statistics: {len(round_budget_stats)} clients")


def calculate_final_results(experiment_result, server, task_start_time, task_end_time,
                           client_total_sm_seconds, client_total_cpu_time, client_total_wall_time,
                           basic_modules, config, fc_net_list, command_queue, result_dict, logger):
    """
    Calculate and summarize final experiment results.
    
    Parameters:
    - experiment_result (dict): Experiment result dictionary (will be updated)
    - server (Server): Server instance
    - task_start_time (float): Task start time
    - task_end_time (float): Task end time
    - client_total_sm_seconds (list): Cumulative GPU SM·seconds (using list as mutable container)
    - client_total_cpu_time (list): Cumulative CPU time (using list as mutable container)
    - client_total_wall_time (list): Cumulative wall-clock time (using list as mutable container)
    - basic_modules (dict): Dictionary of basic modules
    - config (dict): Experiment configuration dictionary
    - fc_net_list (list): List of MIA shadow models
    - command_queue (Queue or None): GPU monitoring command queue
    - result_dict (dict or None): GPU monitoring result dictionary
    - logger: Logger instance
    """
    # Extract actual values from lists
    sm_seconds_value = client_total_sm_seconds[0] if isinstance(client_total_sm_seconds, list) else client_total_sm_seconds
    cpu_time_value = client_total_cpu_time[0] if isinstance(client_total_cpu_time, list) else client_total_cpu_time
    wall_time_value = client_total_wall_time[0] if isinstance(client_total_wall_time, list) else client_total_wall_time
    
    # Calculate and record experiment results
    experiment_result["acc"] = server.evaluate()  # Calculate post-training accuracy
    experiment_result["total_time_cost"] = task_end_time - task_start_time  # Calculate total duration
    experiment_result["client_train_time"] = wall_time_value  # Update client training time as cumulative wall-clock time
    experiment_result["client_cpu_time"] = cpu_time_value  # Add CPU time record
    experiment_result["server_train_time"] = experiment_result["total_time_cost"] - experiment_result["client_train_time"]
    experiment_result["total_communication_size"] /= 8 * 1024 * 1024  # Convert to MB
    
    # Add GPU SM utilization results
    if command_queue and result_dict:
        experiment_result["client_gpu_sm_seconds"] = sm_seconds_value
        logger.info(f'Final GPU SM utilization cumulative: {sm_seconds_value} SM·seconds')
        logger.info(f'Final CPU time cumulative: {cpu_time_value} seconds')
        logger.info(f'Final client training wall-clock time: {wall_time_value} seconds')
    
    experiment_result["mia_acc"] = prepare_and_evaluate_attack(basic_modules, config, server.global_model, fc_net_list)
    
    # Calculate compute budget related statistics
    if "client_compute_budgets" in experiment_result and experiment_result["client_compute_budgets"]:
        total_flops = 0.0
        total_clients_exceeded = 0
        total_epochs = 0
        client_count = 0
        
        # Iterate through all clients across all rounds
        for round_stats in experiment_result["client_compute_budgets"]:
            for client_stats in round_stats:
                total_flops += client_stats.get("current_flops", 0.0)
                if client_stats.get("budget_exceeded", False):
                    total_clients_exceeded += 1
                total_epochs += client_stats.get("epoch_count", 0)
                client_count += 1
        
        # Calculate statistics
        if client_count > 0:
            # Total floating-point operations (converted to billions)
            total_flops_giga = total_flops / 1e9
            
            # Average floating-point operations per client per round
            avg_flops_per_client_per_round_giga = total_flops_giga / client_count
            
            # Budget usage percentage (