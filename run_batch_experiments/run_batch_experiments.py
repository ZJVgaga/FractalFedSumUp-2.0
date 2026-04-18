"""
批量实验运行主程序
"""

import os
import time
import signal
import multiprocessing
from multiprocessing import Pool, Manager
import openpyxl

from run_one_experiment.run_and_get_results.config import *
from xlsx_tools import unmerge_sheet, experiment_progress_xlsx_sheet_rename_map
from get_batch_to_experiment_id_map import get_batch_to_experiment_id_map

# 从新模块中导入
from gpu_monitor import GPUMonitor
from experiment_runner import run_experiment_on_gpu


def callback(pid):
    """任务完成后的回调函数，只记录信息，不尝试kill进程"""
    print(f"[Callback] 任务完成，PID: {pid} 已结束")
    # 注意：不再尝试kill进程，multiprocessing.Pool会自动管理进程生命周期
    # 尝试kill已经结束的进程会导致僵尸进程


def error_callback(e):
    print(f"Task encountered an error: {e}")


# GPU监控器命令处理线程
def gpu_monitor_command_handler(gpu_monitor, command_queue, result_dict, stop_event):
    """处理来自子进程的GPU监控命令"""
    import threading
    import time
    
    print("[GPU监控命令处理器] 线程启动")
    
    while not stop_event.is_set():
        try:
            # 非阻塞获取命令
            try:
                command = command_queue.get_nowait()
                if command:
                    cmd_type, pid, caller_pid = command
                    
                    if cmd_type == 'start':
                        success = gpu_monitor.start(pid)
                        result_key = f"result_{caller_pid}_{pid}_start"
                        result_dict[result_key] = ('start', pid, success, caller_pid)
                        print(f"[GPU监控命令处理器] 处理start({pid}) from PID {caller_pid}: {success}")
                    
                    elif cmd_type == 'end':
                        result = gpu_monitor.end(pid)
                        result_key = f"result_{caller_pid}_{pid}_end"
                        result_dict[result_key] = ('end', pid, result, caller_pid)
                        print(f"[GPU监控命令处理器] 处理end({pid}) from PID {caller_pid}: 返回 {result}")
                    
                    elif cmd_type == 'status':
                        status = {
                            'current_processes': dict(gpu_monitor.current_processes),
                            'accumulating_pids': dict(gpu_monitor.accumulating_pids),
                            'running': gpu_monitor.running
                        }
                        result_key = f"result_{caller_pid}_{pid}_status"
                        result_dict[result_key] = ('status', pid, status, caller_pid)
            except:
                pass  # 队列为空
            
            time.sleep(0.1)  # 短暂休眠避免CPU占用过高
            
        except Exception as e:
            print(f"[GPU监控命令处理器] 异常: {e}")
    
    print("[GPU监控命令处理器] 线程停止")

# 批量实验过程函数    
def run_batch_experiments(experiment_progress_xlsx_path, hyperparam_to_cell_address_map, experiment_region, config_map):
    global config
    global experiment_progress_xlsx_sheet_rename_map
    
    # 打开现有的 Excel 文件并拿到批量实验设置表
    experiment_progress_xlsx = openpyxl.load_workbook(experiment_progress_xlsx_path)
    
    # 用config.get(str)对config当中的config_map中的键进行读取，结果存到名为"结果展示表_实验配置"的config_map对应的键位置上
    # 根据传入的sheet名称选择对应的sheet
    config_sheet = experiment_progress_xlsx["结果展示表_实验配置"]  # 使用传入的experiment_region作为sheet的名字
    
    # 更新sheet中的值
    for key, cell_address in config_map.items():
        value_to_write = config.get(key)  # 从config中获取对应的值
        config_sheet[cell_address] = value_to_write  # 将值写入到指定的单元格地址
    
    # 保存修改后的Excel文件
    experiment_progress_xlsx.save(experiment_progress_xlsx_path)
    
    # 按照每个实验单元格的编号把单个实验归类到所属的批次当中，每一个批次存储一个列表，
    # 列表的每一个元素格式为：实验id,存储该实验id的单元格所在的行，和列
    batch_to_experiment_id_map = get_batch_to_experiment_id_map(experiment_progress_xlsx, experiment_region)
    
    """
    batch_to_experiment_id_map格式:
    {
        1: [(实验1, row1, col1), (实验2, row2, col2)],
        2: [(实验3, row3, col3)],
        ...
    }
    """
    
    # 先删除旧的错误实验
    for batch, experiments in batch_to_experiment_id_map.items():
        if experiments:
            print(f"Processing experiments from {batch}th batch:")
            for experiment_id, id_row, id_col in experiments:
                # 如果存在同名sheet，则删除
                if f"{experiment_id}" in experiment_progress_xlsx.sheetnames:
                    del experiment_progress_xlsx[f"{experiment_id}"]
    
    experiment_progress_xlsx.save(experiment_progress_xlsx_path)
    
    # 打开现有的 Excel 文件并拿到批量实验设置表,并取消所有的合并单元格，
    # 合并单元格的内容将复制到每一个子单元格中
    experiment_progress_xlsx = openpyxl.load_workbook(experiment_progress_xlsx_path)
    experiment_setting_sheet = experiment_progress_xlsx[experiment_progress_xlsx_sheet_rename_map['experiment_setting_sheet']]
    experiment_setting_sheet = unmerge_sheet(experiment_setting_sheet)
    
    # 保存解除合并后的工作簿，确保后续进程能获取到正确的数据
    experiment_progress_xlsx.save(experiment_progress_xlsx_path)
    
    # 按批次处理实验,准备所有实验任务
    all_experiments = []
    for batch, experiments in batch_to_experiment_id_map.items():
        if experiments:
            for experiment_id, id_row, id_col in experiments:
                all_experiments.append((experiment_id, id_row, id_col))

    # 使用 Manager 创建共享资源
    with Manager() as manager:
        # 创建GPU使用计数共享字典
        gpu_usage_count = manager.dict()  # 记录每张 GPU 的使用计数
        lock = manager.Lock()  # 创建锁，用于管理 GPU 状态
        
        # 创建GPU监控通信机制
        command_queue = manager.Queue()  # 用于子进程发送命令
        result_dict = manager.dict()     # 用于返回结果
        stop_event = manager.Event()     # 用于停止命令处理线程
        
        # 创建GPU监控器实例
        print("[主进程] 创建GPU监控器...")
        gpu_monitor = GPUMonitor(update_interval=1.0, enable_multiprocess=False)
        
        # 启动GPU监控器
        if gpu_monitor.start_monitoring():
            print("[主进程] GPU监控器已启动")
        else:
            print("[主进程] GPU监控器启动失败")
        
        # 启动命令处理线程
        import threading
        command_thread = threading.Thread(
            target=gpu_monitor_command_handler,
            args=(gpu_monitor, command_queue, result_dict, stop_event),
            daemon=True
        )
        command_thread.start()
        print("[主进程] GPU监控命令处理线程已启动")
        
        # 将监控器引用添加到config中（通过全局变量或参数传递）
        # 这里我们通过修改实验任务的参数来传递监控器引用
        # 将实验任务打包为 (experiment_id, id_row, id_col, gpu_usage_count, lock, ..., gpu_monitor_dict)
        
        # 添加调试信息
        print(f"[主进程] command_queue类型: {type(command_queue)}, 是否为空: {command_queue.empty() if hasattr(command_queue, 'empty') else 'N/A'}")
        print(f"[主进程] result_dict类型: {type(result_dict)}, 内容: {dict(result_dict)}")
        
        gpu_tasks = [
            (str(exp[0]), int(exp[1]), int(exp[2]), gpu_usage_count, lock, 
             str(experiment_progress_xlsx_path), dict(hyperparam_to_cell_address_map),
             command_queue, result_dict)  # 添加GPU监控通信机制
            for exp in all_experiments
        ]

        # 检查是否有实验任务需要运行
        if not gpu_tasks:
            print("[主进程] 警告：没有实验任务需要运行，可能Excel文件中没有实验或所有实验已完成")
            # 停止命令处理线程
            stop_event.set()
            command_thread.join(timeout=5.0)
            print("[主进程] GPU监控命令处理线程已停止")
            
            # 停止GPU监控器
            if gpu_monitor.stop_monitoring():
                print("[主进程] GPU监控器已停止")
            else:
                print("[主进程] GPU监控器停止失败")
            return  # 直接返回，不创建进程池

        try:
            # 使用多进程池运行实验
            with Pool(processes=len(gpu_tasks)) as pool:
                for task in gpu_tasks:
                    pool.apply_async(
                        func=run_experiment_on_gpu, 
                        args=(task,),
                        callback=callback,       # 当任务完成时调用 
                        error_callback=error_callback,  # 出错时调用 
                    )
                # 等待所有任务完成（否则主进程可能提前退出）
                pool.close()
                pool.join()
                
        finally:
            # 停止命令处理线程
            stop_event.set()
            command_thread.join(timeout=5.0)
            print("[主进程] GPU监控命令处理线程已停止")
            
            # 停止GPU监控器
            if gpu_monitor.stop_monitoring():
                print("[主进程] GPU监控器已停止")
            else:
                print("[主进程] GPU监控器停止失败")


# 配置映射
saved_config_to_excel_map = {
    "Federated_Learning_Config": "A3",
    "Dataset": "B3",
    "client_num": "C3",
    "Model": "D3",
    "batch_size": "A5",
    "alpha": "B5",
    "communication_rounds": "C5",
    "learning_rate": "D5",
}


# 封装所有实验配置为一个字典列表，支持多个实验配置
experiments = [

    {
        "hyperparam_to_cell_address_map": {
            "Federated_Learning_Config": "A2-A13",  # 方法列：FedAvg, FedProx, FedAdam等
            #"dp_mechanism": "B2-B2",  # 已从batch_size改为epsilon
            #"dp_epsilon": "B3-B3",  # 已从batch_size改为epsilon
            "compressed_image_size":  "B2-B13",
            #"images_per_class":
            "Dataset": "C1-E1",  # 数据集列：MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "C2-E13",
        "hyperparameter_experiment":True,
        "experiment_xlsx_path": 'FedSumUp-hyperparameter-size.xlsx'
    },
    # FedSumUp-baselines.xlsx - 基础实验（已更新为epsilon和MNIST/FashionMNIST/CIFAR10）
    {
        "hyperparam_to_cell_address_map": {
            "Federated_Learning_Config": "A2-A6",  # 方法列：FedAvg, FedProx, FedAdam等
            "dp_mechanism": "B2-B2",  # 已从batch_size改为epsilon
            #"dp_epsilon": "B3-B3",  # 已从batch_size改为epsilon
            "compressed_image_size": "B6-B6",
            "images_per_class": "B3-B5",
            "Dataset": "C1-E1",  # 数据集列：MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "C2-E6",
        "experiment_xlsx_path": 'FedSumUp-baselines-new.xlsx'
    },
{
        "hyperparam_to_cell_address_map": {
            "Federated_Learning_Config": "A2-A6",  # 方法列：FedAvg, FedProx, FedAdam等
            "dp_epsilon": "B2-B6",  # 已从batch_size改为epsilon
            #"dp_epsilon": "B3-B3",  # 已从batch_size改为epsilon
            #"compressed_image_size": "B6-B6",
            #"images_per_class": "B3-B5",
            "Dataset": "C1-E1",  # 数据集列：MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "C2-E6",
        "experiment_xlsx_path": 'FedSumUp-FedAvg-DP.xlsx'
    },
    {
        "hyperparam_to_cell_address_map": {
            #"Federated_Learning_Config": "A2-A6",  # 方法列：FedAvg, FedProx, FedAdam等
            #"dp_epsilon": "B2-B6",  # 已从batch_size改为epsilon
            #"dp_epsilon": "B3-B3",  # 已从batch_size改为epsilon
            #"compressed_image_size": "B6-B6",
            #"images_per_class": "B3-B5",
            "alpha":"B1-E1",
            "Federated_Learning_Config": "A2-A4",
            
            #"Dataset": "C1-E1",  # 数据集列：MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "B2-E4",
        "experiment_xlsx_path": 'FedSumUp-NonIID.xlsx'
    },
        {
        "hyperparam_to_cell_address_map": {
            #"Federated_Learning_Config": "A2-A6",  # 方法列：FedAvg, FedProx, FedAdam等
            #"dp_epsilon": "B2-B6",  # 已从batch_size改为epsilon
            #"dp_epsilon": "B3-B3",  # 已从batch_size改为epsilon
            #"compressed_image_size": "B6-B6",
            #"images_per_class": "B3-B5",
            "client_num":"B1-E1",
            "Federated_Learning_Config": "A2-A4",
            
            #"Dataset": "C1-E1",  # 数据集列：MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "B2-E4",
        "experiment_xlsx_path": 'FedSumUp-clientnum.xlsx'
    },


]
{
        "hyperparam_to_cell_address_map": {
            "Federated_Learning_Config": "A2-A6",  # 方法列：FedAvg, FedProx, FedAdam等
            "dp_epsilon": "B2-B6",  # 已从batch_size改为epsilon
            #"dp_epsilon": "B3-B3",  # 已从batch_size改为epsilon
            #"compressed_image_size": "B6-B6",
            #"images_per_class": "B3-B5",
            "Dataset": "C1-E1",  # 数据集列：MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "C2-E6",
        "experiment_xlsx_path": 'FedSumUp-FedAvg-DP.xlsx'
    },
{
        "hyperparam_to_cell_address_map": {
            "Federated_Learning_Config": "A2-A13",  # 方法列：FedAvg, FedProx, FedAdam等
            #"dp_mechanism": "B2-B2",  # 已从batch_size改为epsilon
            #"dp_epsilon": "B3-B3",  # 已从batch_size改为epsilon
            "compressed_image_size":  "B2-B13",
            #"images_per_class":
            "Dataset": "C1-E1",  # 数据集列：MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "C2-E13",
        
        "experiment_xlsx_path": 'FedSumUp-visualize.xlsx'
    },


if __name__ == "__main__":
    # 设置多进程启动方式为 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    # 遍历每个实验配置并运行
    for config in experiments:
        run_batch_experiments(
            config["experiment_xlsx_path"],
            config["hyperparam_to_cell_address_map"],
            config["experiment_region"],
            saved_config_to_excel_map  # 假设这个变量已在外部定义
        )
