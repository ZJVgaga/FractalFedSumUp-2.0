from .run_and_get_results.run_and_get_results import run_and_get_results
from .save_results_to_sheet.save_results_to_sheet import save_results_to_sheet
import random
import os
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
import time
def run_one_experiment(experiment_result_template_sheet,config,gpu_monitor_dict):
    """运行单次实验，并记录日志"""
    logger = setup_logger(f"ex-id_{config.get('experiment_index')}_{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
    random.seed(config.get("seed"))
    all_results = []
    # 重复做N次实验
    for _ in range(config.get("experiment_times")):
        # 获取实验结果
        config.set("seed",random.randint(0, 522222220))#201331926)#
        logger.info(f"seed设置为：{config.get('seed')}")
        result = run_and_get_results(config, logger,gpu_monitor_dict)#设该函数返回一个字典，格式为 {result1: value1, result2: value2, ...}
        if not result:
            logger.error("错误：未获取到实验结果。")
            continue
        all_results.append(result)
    if not all_results:
        logger.error("错误：没有有效的实验结果。")
        return None
    # 总结果字典
    final_result = {}
    # 获取所有键
    all_keys = set().union(*[result.keys() for result in all_results])
    # 遍历所有键
    for key in all_keys:
        processed_value = process_result_values(all_results, key)
        if processed_value is not None:
            final_result[key] = processed_value
        else:
            logger.warning(f"键 '{key}' 的处理结果为 None，跳过")

    # 打印 result_to_cell_map 到日志
    logger.info("多次试验后的实验结果:")
    for key, value in final_result.items():
        logger.info(f"  {key}: {value}")
    


      
    # 创建结果与单元格地址的映射关系
    result_to_cell_map ={
        "acc": 'B5',  #最终准确率
        "total_communication_size":"D5",#总通讯量
        "total_time_cost":'E5', # 总时长
        "client_train_time":'F5',
        "server_train_time":'G5',
        "round_accuracies":"B8-B108",#每轮准确率
        "round_mia_accuracies":"C8-C108",#每轮MIA准确率（新增）
        "mia_acc":"C5",
        "client_gpu_sm_seconds": 'H5',  # GPU SM利用率累计值
        "client_cpu_time": 'I5',  # CPU时间累计值
        # 计算预算相关字段
        "compute_budget_enabled": 'J5',  # 是否启用计算预算
        "compute_budget_per_client_per_round": 'K5',  # 每个客户端每轮的计算预算
        "budget_check_frequency": 'L5',  # 预算检查频率
        "total_flops_giga": 'M5',  # 总浮点运算次数（十亿次）
        "avg_flops_per_client_per_round_giga": 'N5',  # 每个客户端每轮平均浮点运算次数
        "budget_utilization_percent": 'O5',  # 预算使用率百分比
        "clients_exceeded_budget": 'P5',  # 超支预算的客户端数量
        "avg_epochs_per_client": 'Q5',  # 每个客户端平均训练轮数
    }
    try:
        clone_sheet = save_results_to_sheet(config,experiment_result_template_sheet, final_result, result_to_cell_map)
        logger.info("实验结果保存成功。")
        return clone_sheet
    except Exception as e:
        logger.error(f"保存结果失败: {e}", exc_info=True)
        return None

import numpy as np

def calculate_mean_std(values):
    """
    计算均值和标准差，并以 A±B 的字符串形式返回。
    参数:
    - values: 一个包含数值的列表或数组。
    返回:
    - 均值和标准差的字符串表示形式 "A±B"。
    """
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)  # 使用样本标准差
    if std_val is None:
        return  f"{mean_val:.4f}"
    return f"{mean_val:.4f}±{std_val:.4f}"

def process_result_values(results_list, key):
    """
    根据结果值的类型（列表、数字、字典或其他）计算均值和标准差。
    参数:
    - results_list: 包含多次实验结果的列表，每个结果都是字典。
    - key: 当前处理的键。
    返回:
    - 处理后的结果值，格式为 "A±B" 或原始值（如果无法计算均值和标准差）。
    """
    # 检查键是否包含点号（表示嵌套字典）
    if '.' in key:
        # 处理嵌套键，如 "flops_stats.total_flops_giga"
        parts = key.split('.')
        
        # 提取所有结果中该嵌套键的值
        values = []
        for result in results_list:
            value = result
            valid_path = True
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    valid_path = False
                    break
            if valid_path and value is not None:
                values.append(value)
        
        if not values:
            # 如果没有任何结果包含该路径，返回 None
            return None
        
        # 获取第一个有效值用于类型检查
        first_value = values[0]
        
        # 检查值的类型
        if isinstance(first_value, list):
            # 如果是列表，计算每个对应位置元素的均值和标准差
            transposed_lists = zip(*values)
            processed_values = [calculate_mean_std(list(vals)) for vals in transposed_lists]
            return processed_values
        elif isinstance(first_value, (int, float)):
            # 如果是数字，计算均值和标准差
            return calculate_mean_std(values)
        else:
            # 对于其他类型的数据，取第一个结果的值
            return first_value
    else:
        # 处理普通键
        if key not in results_list[0]:
            return None
            
        first_value = results_list[0][key]
        if isinstance(first_value, list):
            # 如果是列表，计算每个对应位置元素的均值和标准差
            transposed_lists = zip(*[result[key] for result in results_list])
            processed_values = [calculate_mean_std(list(vals)) for vals in transposed_lists]
            return processed_values
        elif isinstance(first_value, (int, float)):
            # 如果是数字，计算均值和标准差
            values = [result[key] for result in results_list]
            return calculate_mean_std(values)
        else:
            # 对于其他类型的数据，取第一个结果的值
            return first_value
