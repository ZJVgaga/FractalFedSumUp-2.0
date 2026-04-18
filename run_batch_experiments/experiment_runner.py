"""
实验运行器模块：在GPU上运行单个实验
"""

import os
import time
import traceback
import re
import GPUtil
from filelock import FileLock
import openpyxl
from run_one_experiment.run_and_get_results.config import *
from xlsx_tools import unmerge_sheet, experiment_progress_xlsx_sheet_rename_map
from generate_config_for_one_experiment.generate_config_for_one_experiment import generate_config_for_one_experiment
from run_one_experiment.run_one_experiment import run_one_experiment


def run_experiment_on_gpu(args):
    """
    在GPU上运行单个实验
    
    Args:
        args: 包含以下元素的元组：
            - experiment_id: 实验ID
            - id_row: 实验ID所在行
            - id_col: 实验ID所在列
            - gpu_usage_count: GPU使用计数共享字典
            - lock: 进程锁
            - experiment_progress_xlsx_path: 实验进度Excel文件路径
            - hyperparam_to_cell_address_map: 超参数到单元格地址的映射
            - command_queue: GPU监控命令队列
            - result_dict: GPU监控结果字典
    
    Returns:
        pid: 当前进程的PID
    """
    pid = os.getpid()
    
    # 解包参数，现在包含command_queue和result_dict
    experiment_id, id_row, id_col, gpu_usage_count, lock, experiment_progress_xlsx_path, hyperparam_to_cell_address_map, command_queue, result_dict = args
    
    # 打印当前进程的PID，方便调试
    print(f"[实验进程 {experiment_id}] PID: {pid}, 可以访问GPU监控通信机制")
    
    # 选择可用的GPU
    gpu_id = None
    while True:
        # 检测所有 GPU 的空闲状态
        gpus = GPUtil.getGPUs()
        available_gpus = [
            gpu.id for gpu in gpus 
            if gpu.load <= 0.5 and gpu.memoryUtil <= 0.9 and gpu.id in [0, 1, 2, 3, 4, 5, 6, 7]  # GPU-Util < 30% 且 Memory-Usage < 50%
        ]
        
        with lock:  # 使用锁确保线程安全
            # 过滤出未被其他进程占用的 GPU
            truly_available_gpus = [
                gpu_id for gpu_id in available_gpus 
                if gpu_usage_count.get(gpu_id, 0) < 1  # max_processes_per_gpu
            ]
            
            if truly_available_gpus:
                # 选择第一张真正空闲的 GPU
                gpu_id = truly_available_gpus[0]
                
                # 更新 GPU 使用计数
                gpu_usage_count[gpu_id] = gpu_usage_count.get(gpu_id, 0) + 1
                print(f"Experiment {experiment_id} is assigned to GPU {gpu_id} (shared)")
                break
            else:
                # 如果没有真正空闲的 GPU，则等待 1 秒后重试
                print(f"No available GPUs for experiment {experiment_id}. Waiting...")
                time.sleep(1)
                continue
    
    try:
        # 定义文件锁路径
        lock_file = experiment_progress_xlsx_path + ".lock"
        
        # 加载实验进度表
        with FileLock(lock_file):  # 确保只有一个进程能进入这个块
            # 每次操作都重新加载工作簿，确保获取最新状态
            wb = openpyxl.load_workbook(experiment_progress_xlsx_path)
            experiment_setting_sheet = wb[experiment_progress_xlsx_sheet_rename_map['experiment_setting_sheet']]
            # 确保工作表已解除合并
            experiment_setting_sheet = unmerge_sheet(experiment_setting_sheet)
            
        # 生成实验配置
        config = ConfigTree()
        config = generate_config_for_one_experiment(
            experiment_setting_sheet, hyperparam_to_cell_address_map, experiment_id, id_row, id_col, config
        )
        
        # 将GPU监控通信机制添加到config中
        # 使用一个特殊的键来存储，这样其他模块可以通过config访问
        gpu_monitor_dict = {
            'command_queue': command_queue,
            'result_dict': result_dict,
            'process_pid': pid
        }

        print(f"[实验进程 {experiment_id}] GPU监控通信机制已添加到config中")
        
        # 强制使用GPU，如果CUDA不可用则抛出错误
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA is not available for experiment {experiment_id}. GPU required.")
            
            # 验证GPU ID是否有效
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f"Invalid GPU ID {gpu_id}. Available GPUs: {torch.cuda.device_count()}")
            
            config.set("device", f"cuda:{gpu_id}")
            print(f"Experiment {experiment_id} using GPU {gpu_id}")
            
        except Exception as e:
            # 如果出现任何错误，直接抛出，不允许回退到CPU
            raise RuntimeError(f"Failed to set GPU device for experiment {experiment_id}: {e}")
        
        # 运行实验并获取完成状态
        result_ws = run_one_experiment(
            wb[experiment_progress_xlsx_sheet_rename_map["one_experiment_result_template_sheet"]],
            config,
            gpu_monitor_dict
        )
            
        # 将实验结果保存到experiment_progress_xlsx中，命名为 experiment_id
        if result_ws is not None:
            # 确保每次保存时都重新加载工作簿以避免数据丢失
            with FileLock(lock_file):
                wb = openpyxl.load_workbook(experiment_progress_xlsx_path)
                # 重新获取experiment_setting_sheet，确保使用新加载的工作簿
                experiment_setting_sheet = wb[experiment_progress_xlsx_sheet_rename_map['experiment_setting_sheet']]
                
                # 检查是否存在同名的工作表，如果存在则删除旧的工作表
                if f"{experiment_id}" in wb.sheetnames:
                    del wb[f"{experiment_id}"]
                
                # 创建新的工作表并将实验结果复制进去
                new_ws = wb.create_sheet(title=f"{experiment_id}")
                
                # 将 result_ws 的内容复制到新工作表中
                for row in result_ws.iter_rows():
                    new_row = []
                    for cell in row:
                        new_row.append(cell.value)
                    new_ws.append(new_row)
                
                # 将实验编号从"实验ID-批次"改为"实验ID-0"，表示已完成
                cell = experiment_setting_sheet.cell(row=id_row, column=id_col)
                cell_value = str(cell.value).strip()
                
                # 解析实验编号格式：支持"实验ID-批次"或纯数字
                match = re.match(r'^(\d+)(?:-(\d+))?$', cell_value)
                if match:
                    experiment_id_num = match.group(1)
                    # 更新为"实验ID-0"格式
                    cell.value = f"{experiment_id_num}-0"
                    print(f"实验 {experiment_id_num} 已完成，单元格值更新为: {cell.value}")
                else:
                    print(f"警告：无法解析实验编号格式: {cell_value}")
                
                wb.save(experiment_progress_xlsx_path)
            
            print(f"Experiment {experiment_id} processed and results saved to {experiment_progress_xlsx_path}")
            print(f"Experiment {experiment_id} FINISHED on GPU {gpu_id}")
                
        else:
            raise RuntimeError(f"Experiment {experiment_id} did not complete successfully")
    
    except Exception as e:
        print("子进程发生错误:")
        traceback.print_exc()  # 打印完整的堆栈跟踪
        raise
        
    finally:
        # 无论成功与否都释放 GPU
        with lock:
            if gpu_id in gpu_usage_count:
                gpu_usage_count[gpu_id] -= 1  # 减少 GPU 使用计数
        print(f"GPU {gpu_id} released by experiment {experiment_id}")
    
    return pid  # 返回当前进程的 PID
