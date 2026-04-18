import os
import time

from .attack.mia.Train_Logits_Inference_Model.MIA import train_inference_model, prepare_and_evaluate_attack, load_fc_net_list, save_fc_net_list
from .process_tools import measure_size, early_stopper
from .init_basic_modules import init_basic_modules


def initialize_experiment_components(config, logger):
    """
    初始化实验所需的基本组件（服务器和客户端）。
    
    参数：
    - config (dict): 实验配置字典，包含client_num等配置
    - logger: 日志记录器
    
    返回：
    - tuple: (basic_modules, server, client_list)
      - basic_modules (dict): 基础模块字典
      - server (Server): 服务器实例
      - client_list (list): 客户端实例列表，格式为 [Client1, Client2, ...]
    """
    # 首先初始化需要的基本模块
    basic_modules, Server, Client = init_basic_modules(config, logger)
    
    # 初始化客户端列表（同时初始化client_num个客户端）
    client_list = []
    for i in range(config.get("client_num")):
        client_i = Client(basic_modules, config, logger, i)
        client_list.append(client_i)
    
    # 初始化Server
    server = Server(basic_modules, config, logger, client_list)
    logger.info('Server and Clients have been created.')
    
    return basic_modules, server, client_list


def create_experiment_result_structure():
    """
    创建并初始化实验结果数据结构。
    
    返回：
    - dict: 初始化的实验结果字典，包含以下字段：
      - total_communication_size (float): 总通信大小（字节）
      - client_train_time (float): 客户端训练总时间（秒）
      - round_accuracies (list): 每轮准确率列表，格式为 [acc1, acc2, ...]
      - round_mia_accuracies (list): 每轮MIA准确率列表，格式为 [mia_acc1, mia_acc2, ...]
      - client_compute_budgets (list): 每轮客户端计算预算使用情况列表，
        格式为 [[{client_stats1}, {client_stats2}, ...], ...]
    """
    return {
        "total_communication_size": 0,
        "client_train_time": 0,
        "round_accuracies": [],
        "round_mia_accuracies": [],  # 新增：每轮MIA准确率
        "client_compute_budgets": [],  # 新增：每轮客户端计算预算使用情况
    }


def load_or_train_mia_model(basic_modules, config, logger):
    """
    加载或训练MIA（成员推理攻击）影子模型。
    
    参数：
    - basic_modules (dict): 基础模块字典
    - config (dict): 实验配置字典
    - logger: 日志记录器
    
    返回：
    - list or None: MIA影子模型列表，格式为 [fc_net1, fc_net2, ...]
    """
    logger.info('开始加载MIA影子模型...')
    fc_net_list = load_fc_net_list(config)  # 尝试加载已有模型
    
    # 这个地方后期添加MIA的检测，如果不是MIA攻击方法，就不需要训练影子模型
    # 训练影子模型
    if fc_net_list is None: 
        logger.info('MIA影子模型不存在，开始训练影子模型...')
        fc_net_list = train_inference_model(basic_modules, config)
        logger.info('MIA影子模型训练完成，开始保存...')
        save_fc_net_list(fc_net_list, config)
        logger.info('MIA影子模型保存完成')
    else:
        logger.info('MIA影子模型加载成功')
    
    return fc_net_list


def setup_gpu_monitoring(gpu_monitor_dict, logger):
    """
    设置GPU监控相关参数。
    
    参数：
    - gpu_monitor_dict (dict or None): GPU监控通信字典，包含command_queue和result_dict
    - logger: 日志记录器
    
    返回：
    - tuple: (command_queue, result_dict, process_pid, client_total_sm_seconds, 
              client_total_cpu_time, client_total_wall_time)
      - command_queue (Queue or None): 命令队列
      - result_dict (dict or None): 结果字典
      - process_pid (int): 进程PID
      - client_total_sm_seconds (list): 累计所有客户端的GPU SM·秒（使用列表作为可变容器）
      - client_total_cpu_time (list): 累计所有客户端的CPU时间（使用列表作为可变容器）
      - client_total_wall_time (list): 累计所有客户端的墙上时钟时间（使用列表作为可变容器）
    """
    # 初始化GPU监控通信，使用列表作为可变容器
    client_total_sm_seconds = [0.0]  # 累计所有客户端的GPU SM·秒
    client_total_cpu_time = [0.0]    # 累计所有客户端的CPU时间
    client_total_wall_time = [0.0]   # 累计所有客户端的墙上时钟时间
    
    # 如果提供了GPU监控字典，则使用它来测量GPU SM利用率
    if gpu_monitor_dict:
        logger.info(f'gpu_monitor_dict存在: {gpu_monitor_dict}')
        command_queue = gpu_monitor_dict.get('command_queue')
        result_dict = gpu_monitor_dict.get('result_dict')
        process_pid = gpu_monitor_dict.get('process_pid', os.getpid())
        
        # 检查command_queue和result_dict是否有效
        if command_queue and result_dict:
            logger.info(f'启用GPU SM利用率监控，进程PID: {process_pid}')
            logger.info(f'command_queue类型: {type(command_queue)}')
            logger.info(f'result_dict类型: {type(result_dict)}')
        else:
            command_queue = None
            result_dict = None
            process_pid = os.getpid()
            logger.info('GPU监控字典存在但command_queue或result_dict无效')
    else:
        command_queue = None
        result_dict = None
        process_pid = os.getpid()
        logger.info('未启用GPU SM利用率监控（gpu_monitor_dict不存在）')
    
    return command_queue, result_dict, process_pid, client_total_sm_seconds, client_total_cpu_time, client_total_wall_time


def get_actual_rounds(basic_modules, config, logger):
    """
    获取实际的训练轮次数。
    
    参数：
    - basic_modules (dict): 基础模块字典
    - config (dict): 实验配置字典
    - logger: 日志记录器
    
    返回：
    - int: 实际的训练轮次数
    """
    # 对于FedSumUp，新版本每个round只需要一次通信
    fed_strategy = config.get("Federated_Learning_Config")
    if fed_strategy == "FedSumUp":
        actual_rounds = basic_modules["communication_rounds"]
        logger.info(f"FedSumUp策略（新版本）：每个round只需要一次通信，轮次数：{actual_rounds}")
    else:
        actual_rounds = basic_modules["communication_rounds"]
    
    return actual_rounds


def measure_gpu_utilization(command_queue, result_dict, process_pid, operation_type, logger):
    """
    测量GPU利用率。
    
    参数：
    - command_queue (Queue or None): GPU监控命令队列
    - result_dict (dict or None): GPU监控结果字典
    - process_pid (int): 进程PID
    - operation_type (str): 操作类型，'start'或'end'
    - logger: 日志记录器
    
    返回：
    - float: GPU SM利用率累计值（仅当operation_type为'end'时返回）
    """
    if not command_queue or not result_dict:
        logger.debug(f"GPU监控未启用，跳过{operation_type}操作")
        return 0.0
    
    # 发送命令
    command_queue.put((operation_type, process_pid, os.getpid()))
    logger.info(f'发送GPU监控{operation_type}命令，PID: {process_pid}')
    
    # 等待结果
    start_time = time.time()
    sm_accumulated = 0.0
    
    while time.time() - start_time < 5.0:  # 最多等待5秒
        result_key = f"result_{os.getpid()}_{process_pid}_{operation_type}"
        if result_key in result_dict:
            result = result_dict[result_key]
            if operation_type == 'end' and len(result) >= 3:
                sm_accumulated = result[2]  # 获取累计值
                logger.info(f'收到GPU监控{operation_type}结果: {sm_accumulated}')
            elif operation_type == 'start':
                logger.info(f'收到GPU监控{operation_type}结果: {result}')
            break
        time.sleep(0.1)
    
    return sm_accumulated


def process_client_training(client, server, command_queue, result_dict, process_pid, 
                           client_total_sm_seconds, client_total_cpu_time, client_total_wall_time,
                           experiment_result, config, logger):
    """
    处理单个客户端的训练过程，包括GPU监控。
    
    参数：
    - client (Client): 客户端实例
    - server (Server): 服务器实例
    - command_queue (Queue or None): GPU监控命令队列
    - result_dict (dict or None): GPU监控结果字典
    - process_pid (int): 进程PID
    - client_total_sm_seconds (list): 累计GPU SM·秒（使用列表作为可变容器）
    - client_total_cpu_time (list): 累计CPU时间（使用列表作为可变容器）
    - client_total_wall_time (list): 累计墙上时钟时间（使用列表作为可变容器）
    - experiment_result (dict): 实验结果字典（将被更新）
    - config (dict): 实验配置字典
    - logger: 日志记录器
    
    返回：
    - dict: 客户端返回的数据，格式为 {data_key: data_value, ...}
    """
    # 用户可自定义的过程(开始)
    server_data = server.arrange_server_data_to_client()
    client.receive_data_from_server(server_data)
    
    # 1. 使用封装的GPU测算函数
    # 开始GPU监控
    measure_gpu_utilization(command_queue, result_dict, process_pid, 'start', logger)
    
    # 记录CPU时间开始
    cpu_start_time = time.process_time()
    # 记录墙上时钟时间开始
    wall_start_time = time.time()
    
    # 2. 执行客户端处理
    # 注意：client.process()内部已经处理计算预算，不需要传入参数
    client.process()
    
    # 记录CPU时间结束
    cpu_end_time = time.process_time()
    # 记录墙上时钟时间结束
    wall_end_time = time.time()
    
    # 更新累计时间（通过列表索引更新）
    client_cpu_time_delta = cpu_end_time - cpu_start_time
    client_total_cpu_time[0] += client_cpu_time_delta
    
    client_wall_time_delta = wall_end_time - wall_start_time
    client_total_wall_time[0] += client_wall_time_delta
    
    # 结束GPU监控并获取累计值
    sm_accumulated = measure_gpu_utilization(command_queue, result_dict, process_pid, 'end', logger)
    client_total_sm_seconds[0] += sm_accumulated
    
    # 记录详细的CPU时间信息到日志
    logger.info(f"客户端 {client.client_id if hasattr(client, 'client_id') else 'unknown'} CPU时间: {client_cpu_time_delta:.4f}秒, 累计CPU时间: {client_total_cpu_time[0]:.4f}秒")
    
    # 3. 获取客户端返回的数据
    received_data = client.send_data_to_server()
    
    # 4. 计算从客户端到服务器的信息大小，用于评估通信开销
    experiment_result["total_communication_size"] += (measure_size(server_data) + measure_size(received_data))
    
    return received_data


def process_server_round(server, received_data_list, basic_modules, config, 
                        fc_net_list, experiment_result, rounds, logger):
    """
    处理服务器端的聚合和更新。
    
    参数：
    - server (Server): 服务器实例
    - received_data_list (list): 客户端返回的数据列表，格式为 [{data1}, {data2}, ...]
    - basic_modules (dict): 基础模块字典
    - config (dict): 实验配置字典
    - fc_net_list (list): MIA影子模型列表
    - experiment_result (dict): 实验结果字典（将被更新）
    - rounds (int): 当前轮次
    - logger: 日志记录器
    """
    logger.info('---------- server process ----------')
    # 用户可自定义的过程(开始)
    merged_data = server.merge_data(received_data_list)
    server.process(merged_data)
    
    # 计算并记录本轮MIA准确率
    mia_accuracy = prepare_and_evaluate_attack(basic_modules, config, server.global_model, fc_net_list)
    logger.info(f"MIA攻击准确率为:{mia_accuracy}")
    experiment_result["round_mia_accuracies"].append(mia_accuracy)  # 记录每轮MIA准确率
    
    # 收集本轮客户端的计算预算使用情况
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
    logger.info(f"第{rounds}轮客户端计算预算统计: {len(round_budget_stats)}个客户端")


def calculate_final_results(experiment_result, server, task_start_time, task_end_time,
                           client_total_sm_seconds, client_total_cpu_time, client_total_wall_time,
                           basic_modules, config, fc_net_list, command_queue, result_dict, logger):
    """
    计算和汇总最终实验结果。
    
    参数：
    - experiment_result (dict): 实验结果字典（将被更新）
    - server (Server): 服务器实例
    - task_start_time (float): 任务开始时间
    - task_end_time (float): 任务结束时间
    - client_total_sm_seconds (list): 累计GPU SM·秒（使用列表作为可变容器）
    - client_total_cpu_time (list): 累计CPU时间（使用列表作为可变容器）
    - client_total_wall_time (list): 累计墙上时钟时间（使用列表作为可变容器）
    - basic_modules (dict): 基础模块字典
    - config (dict): 实验配置字典
    - fc_net_list (list): MIA影子模型列表
    - command_queue (Queue or None): GPU监控命令队列
    - result_dict (dict or None): GPU监控结果字典
    - logger: 日志记录器
    """
    # 从列表中提取实际值
    sm_seconds_value = client_total_sm_seconds[0] if isinstance(client_total_sm_seconds, list) else client_total_sm_seconds
    cpu_time_value = client_total_cpu_time[0] if isinstance(client_total_cpu_time, list) else client_total_cpu_time
    wall_time_value = client_total_wall_time[0] if isinstance(client_total_wall_time, list) else client_total_wall_time
    
    # 计算和记录实验结果
    experiment_result["acc"] = server.evaluate()  # 计算训练后准确率
    experiment_result["total_time_cost"] = task_end_time - task_start_time  # 计算总时长
    experiment_result["client_train_time"] = wall_time_value  # 更新客户端训练时间为累计的墙上时钟时间
    experiment_result["client_cpu_time"] = cpu_time_value  # 添加CPU时间记录
    experiment_result["server_train_time"] = experiment_result["total_time_cost"] - experiment_result["client_train_time"]
    experiment_result["total_communication_size"] /= 8 * 1024 * 1024  # 转换为MB
    
    # 添加GPU SM利用率结果
    if command_queue and result_dict:
        experiment_result["client_gpu_sm_seconds"] = sm_seconds_value
        logger.info(f'最终GPU SM利用率累计: {sm_seconds_value} SM·秒')
        logger.info(f'最终CPU时间累计: {cpu_time_value} 秒')
        logger.info(f'最终客户端训练墙上时钟时间: {wall_time_value} 秒')
    
    experiment_result["mia_acc"] = prepare_and_evaluate_attack(basic_modules, config, server.global_model, fc_net_list)
    
    # 计算计算预算相关的统计数据
    if "client_compute_budgets" in experiment_result and experiment_result["client_compute_budgets"]:
        total_flops = 0.0
        total_clients_exceeded = 0
        total_epochs = 0
        client_count = 0
        
        # 遍历所有轮次的所有客户端
        for round_stats in experiment_result["client_compute_budgets"]:
            for client_stats in round_stats:
                total_flops += client_stats.get("current_flops", 0.0)
                if client_stats.get("budget_exceeded", False):
                    total_clients_exceeded += 1
                total_epochs += client_stats.get("epoch_count", 0)
                client_count += 1
        
        # 计算统计数据
        if client_count > 0:
            # 总浮点运算次数（转换为十亿次）
            total_flops_giga = total_flops / 1e9
            
            # 每个客户端每轮平均浮点运算次数
            avg_flops_per_client_per_round_giga = total_flops_giga / client_count
            
            # 预算使用率百分比（平均）
            total_budget_usage = 0.0
            for round_stats in experiment_result["client_compute_budgets"]:
                for client_stats in round_stats:
                    total_budget_usage += client_stats.get("budget_usage_percent", 0.0)
            budget_utilization_percent = total_budget_usage / client_count if client_count > 0 else 0.0
            
            # 每个客户端平均训练轮数
            avg_epochs_per_client = total_epochs / client_count
            
            # 添加计算预算统计数据到实验结果
            experiment_result["compute_budget_enabled"] = config.get("enable_compute_budget", True)
            experiment_result["compute_budget_per_client_per_round"] = config.get("compute_budget_per_client_per_round", 1e9)
            experiment_result["budget_check_frequency"] = config.get("budget_check_frequency", 1)
            experiment_result["total_flops_giga"] = total_flops_giga
            experiment_result["avg_flops_per_client_per_round_giga"] = avg_flops_per_client_per_round_giga
            experiment_result["budget_utilization_percent"] = budget_utilization_percent
            experiment_result["clients_exceeded_budget"] = total_clients_exceeded
            experiment_result["avg_epochs_per_client"] = avg_epochs_per_client
            
            logger.info(f"计算预算统计数据:")
            logger.info(f"  总浮点运算次数: {total_flops_giga:.2f} GFLOPs")
            logger.info(f"  每个客户端每轮平均: {avg_flops_per_client_per_round_giga:.2f} GFLOPs")
            logger.info(f"  预算使用率: {budget_utilization_percent:.1f}%")
            logger.info(f"  超支预算的客户端数量: {total_clients_exceeded}")
            logger.info(f"  每个客户端平均训练轮数: {avg_epochs_per_client:.1f}")
        else:
            logger.warning("没有收集到客户端计算预算数据")
    else:
        logger.warning("未启用计算预算或未收集到计算预算数据")


def run_and_get_results(config, logger, gpu_monitor_dict=None):
    """
    模拟运行实验并获取实验结果。
    
    参数：
    - config (dict): 实验配置字典
    - logger: 日志记录器
    - gpu_monitor_dict (dict or None): GPU监控通信字典，包含command_queue和result_dict
    
    返回：
    - dict: 返回实验结果的字典，包含以下字段：
      - total_communication_size (float): 总通信大小（MB）
      - client_train_time (float): 客户端训练总时间（秒）
      - round_accuracies (list): 每轮准确率列表，格式为 [acc1, acc2, ...]
      - round_mia_accuracies (list): 每轮MIA准确率列表，格式为 [mia_acc1, mia_acc2, ...]
      - client_compute_budgets (list): 每轮客户端计算预算使用情况列表，
        格式为 [[{client_stats1}, {client_stats2}, ...], ...]
      - acc (float): 最终准确率
      - total_time_cost (float): 总训练时间（秒）
      - client_cpu_time (float): 客户端CPU时间（秒）
      - server_train_time (float): 服务器训练时间（秒）
      - mia_acc (float): 最终MIA准确率
      - client_gpu_sm_seconds (float, optional): GPU SM利用率累计值（如果启用GPU监控）
      - compute_budget_enabled (bool, optional): 是否启用计算预算
      - total_flops_giga (float, optional): 总浮点运算次数（GFLOPs）
      - avg_flops_per_client_per_round_giga (float, optional): 每个客户端每轮平均浮点运算次数（GFLOPs）
      - budget_utilization_percent (float, optional): 预算使用率百分比
      - clients_exceeded_budget (int, optional): 超支预算的客户端数量
      - avg_epochs_per_client (float, optional): 每个客户端平均训练轮数
    """
    # 1. 初始化实验组件
    basic_modules, server, client_list = initialize_experiment_components(config, logger)
    
    # 2. 创建实验结果结构
    experiment_result = create_experiment_result_structure()
    
    # 3. 加载或训练MIA影子模型
    fc_net_list = load_or_train_mia_model(basic_modules, config, logger)
    
    # 4. 初始化实验的参数记录
    logger.info('初始化实验参数记录...')
    should_early_stop = early_stopper(patience=1)
    task_start_time = time.time()
    
    # 5. 设置GPU监控
    command_queue, result_dict, process_pid, client_total_sm_seconds, client_total_cpu_time, client_total_wall_time = \
        setup_gpu_monitoring(gpu_monitor_dict, logger)
    
    logger.info('开始联邦学习训练轮次...')
    
    # 6. 获取实际的训练轮次数
    actual_rounds = get_actual_rounds(basic_modules, config, logger)
    
    # 7. 对每一轮训练
    for rounds in range(actual_rounds):
        # 7.1 评估当前准确率
        acc = server.evaluate()  # 每隔一轮测一下准确率
        experiment_result["round_accuracies"].append(acc)  # 将当前轮次的准确率添加到列表中
        
        # 7.2 早停判定
        early_stop_triggered = False
        if should_early_stop(rounds, acc, experiment_result["round_accuracies"]):
            logger.info(f"早停在第 {rounds} 轮触发")
            early_stop_triggered = True
            
            
            break
        
        round_start_time = time.time()
        logger.info(f' ====== round {rounds} ======')
        logger.info(f'round {rounds} evaluation: test acc is {acc}')
        logger.info('---------- client process----------')
        
        # 7.3 初始化每一轮接受的所有客户端的数据
        received_data_list = []
        
        # 7.4 处理每个选中的客户端
        for client in server.select_clients():
            # 处理客户端训练
            received_data = process_client_training(
                client, server, command_queue, result_dict, process_pid,
                client_total_sm_seconds, client_total_cpu_time, client_total_wall_time,
                experiment_result, config, logger
            )
            received_data_list.append(received_data)
        
        # 7.5 处理服务器端
        process_server_round(
            server, received_data_list, basic_modules, config,
            fc_net_list, experiment_result, rounds, logger
        )
        
        # 7.6 记录本轮时间
        logger.info(f'total time = {time.time() - round_start_time}')
    
    task_end_time = time.time()
    
    # 8. 在训练结束后（无论是正常完成还是早停）触发PDF生成
    logger.info("训练结束，开始生成可视化汇总PDF...")
    try:
        # 导入可视化模块
        from .visualization_system.image_tracker import ImageTracker
        from .visualization_system.client_visualizer import ClientVisualizer
        
        # 收集所有客户端的数据文件路径（从JPG数据目录收集元数据JSON文件）
        all_client_data_paths = []
        # 直接从配置中获取数据文件路径
        base_dir = config.get("visualization_save_dir", "./fedsumup_visualizations")
        dataset_name = config.get("Dataset", "CIFAR10")
        compressed_size = config.get("compressed_image_size", 24)
        
        # JPG数据目录
        jpg_data_dir = os.path.join(
            base_dir,
            f"fedsumup_visualization_{dataset_name}",
            f"compressed_{compressed_size}",
            "jpg_data"
        )
        
        # 检查目录是否存在
        if os.path.exists(jpg_data_dir):
            # 收集该目录下的所有元数据JSON文件
            for filename in os.listdir(jpg_data_dir):
                if filename.endswith("_metadata.json"):
                    data_path = os.path.join(jpg_data_dir, filename)
                    all_client_data_paths.append(data_path)
        
        logger.info(f"总共收集到 {len(all_client_data_paths)} 个数据文件（来自JPG数据目录）")
        
        # 如果有数据文件，生成汇总PDF
        if all_client_data_paths:
            # 创建ImageTracker（使用第一个客户端的数据）
            if client_list and len(client_list) > 0:
                first_client = client_list[0]
                train_indices = first_client.client_modules.get('target_train_indices', [])
                train_labels = first_client.client_modules.get('target_train_labels', [])
                
                if len(train_indices) == 0 or len(train_labels) == 0:
                    logger.warning("client_modules中没有找到target_train_indices或target_train_labels，使用虚拟数据")
                    train_indices = list(range(len(first_client.client_indices)))
                    train_labels = [first_client.dst_train[idx][1] for idx in first_client.client_indices]
                
                image_tracker = ImageTracker(
                    config, 
                    first_client.dataset_info, 
                    train_indices, 
                    train_labels
                )
                
                # 创建ClientVisualizer
                client_visualizer = ClientVisualizer(image_tracker, config)
                
                # 生成汇总PDF（使用实际轮次数，如果rounds未定义则使用0）
                total_rounds_for_pdf = rounds if 'rounds' in locals() else 0
                pdf_path = client_visualizer.generate_summary_pdf(total_rounds_for_pdf, all_client_data_paths)
                
                if pdf_path:
                    logger.info(f"训练结束后汇总PDF生成成功: {pdf_path}")
                else:
                    logger.warning("训练结束后汇总PDF生成失败")
            else:
                logger.warning("没有可用的客户端来创建ImageTracker")
        else:
            logger.warning("没有收集到任何客户端数据文件，无法生成PDF")
            
    except Exception as e:
        logger.warning(f"训练结束后触发PDF生成失败: {str(e)}")
        import traceback
        logger.debug(f"详细错误信息: {traceback.format_exc()}")
    
    # 9. 计算最终实验结果
    calculate_final_results(
        experiment_result, server, task_start_time, task_end_time,
        client_total_sm_seconds, client_total_cpu_time, client_total_wall_time,
        basic_modules, config, fc_net_list, command_queue, result_dict, logger
    )
    
    return experiment_result


import logging
def setup_logger(name, log_dir="experiment_logger", level=logging.INFO):
    """设置并返回一个 logger 对象，日志文件保存在 ./experiment_logger/ 目录下"""
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 输出到文件（存放在 ./experiment_logger/ 目录下）
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    # 测试用例 3: 更新 fedprox_mu 的值
    from config import config
    
    logger = setup_logger(f"ex-id_temptest_{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
    result = run_and_get_results(config, logger)
    print(result)
