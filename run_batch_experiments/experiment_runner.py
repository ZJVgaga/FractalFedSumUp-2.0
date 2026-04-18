"""
Experiment Runner Module: Run a single experiment on GPU
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
    Run a single experiment on GPU
    
    Args:
        args: A tuple containing the following elements:
            - experiment_id: Experiment ID
            - id_row: Row number of the experiment ID
            - id_col: Column number of the experiment ID
            - gpu_usage_count: Shared dictionary for GPU usage count
            - lock: Process lock
            - experiment_progress_xlsx_path: Path to the experiment progress Excel file
            - hyperparam_to_cell_address_map: Mapping from hyperparameters to cell addresses
            - command_queue: GPU monitoring command queue
            - result_dict: GPU monitoring result dictionary
    
    Returns:
        pid: The PID of the current process
    """
    pid = os.getpid()
    
    # Unpack arguments, now includes command_queue and result_dict
    experiment_id, id_row, id_col, gpu_usage_count, lock, experiment_progress_xlsx_path, hyperparam_to_cell_address_map, command_queue, result_dict = args
    
    # Print the current process PID for debugging
    print(f"[Experiment Process {experiment_id}] PID: {pid}, can access GPU monitoring communication mechanism")
    
    # Select an available GPU
    gpu_id = None
    while True:
        # Detect idle status of all GPUs
        gpus = GPUtil.getGPUs()
        available_gpus = [
            gpu.id for gpu in gpus 
            if gpu.load <= 0.5 and gpu.memoryUtil <= 0.9 and gpu.id in [0, 1, 2, 3, 4, 5, 6, 7]  # GPU-Util < 30% and Memory-Usage < 50%
        ]
        
        with lock:  # Use lock to ensure thread safety
            # Filter out GPUs not occupied by other processes
            truly_available_gpus = [
                gpu_id for gpu_id in available_gpus 
                if gpu_usage_count.get(gpu_id, 0) < 1  # max_processes_per_gpu
            ]
            
            if truly_available_gpus:
                # Select the first truly available GPU
                gpu_id = truly_available_gpus[0]
                
                # Update GPU usage count
                gpu_usage_count[gpu_id] = gpu_usage_count.get(gpu_id, 0) + 1
                print(f"Experiment {experiment_id} is assigned to GPU {gpu_id} (shared)")
                break
            else:
                # If no truly available GPU, wait 1 second and retry
                print(f"No available GPUs for experiment {experiment_id}. Waiting...")
                time.sleep(1)
                continue
    
    try:
        # Define file lock path
        lock_file = experiment_progress_xlsx_path + ".lock"
        
        # Load experiment progress sheet
        with FileLock(lock_file):  # Ensure only one process can enter this block
            # Reload workbook each time to get the latest state
            wb = openpyxl.load_workbook(experiment_progress_xlsx_path)
            experiment_setting_sheet = wb[experiment_progress_xlsx_sheet_rename_map['experiment_setting_sheet']]
            # Ensure the sheet is unmerged
            experiment_setting_sheet = unmerge_sheet(experiment_setting_sheet)
            
        # Generate experiment configuration
        config = ConfigTree()
        config = generate_config_for_one_experiment(
            experiment_setting_sheet, hyperparam_to_cell_address_map, experiment_id, id_row, id_col, config
        )
        
        # Add GPU monitoring communication mechanism to config
        # Use a special key to store it so other modules can access via config
        gpu_monitor_dict = {
            'command_queue': command_queue,
            'result_dict': result_dict,
            'process_pid': pid
        }

        print(f"[Experiment Process {experiment_id}] GPU monitoring communication mechanism added to config")
        
        # Force GPU usage, raise error if CUDA is not available
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA is not available for experiment {experiment_id}. GPU required.")
            
            # Validate GPU ID
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f"Invalid GPU ID {gpu_id}. Available GPUs: {torch.cuda.device_count()}")
            
            config.set("device", f"cuda:{gpu_id}")
            print(f"Experiment {experiment_id} using GPU {gpu_id}")
            
        except Exception as e:
            # If any error occurs, raise directly, no fallback to CPU allowed
            raise RuntimeError(f"Failed to set GPU device for experiment {experiment_id}: {e}")
        
        # Run experiment and get completion status
        result_ws = run_one_experiment(
            wb[experiment_progress_xlsx_sheet_rename_map["one_experiment_result_template_sheet"]],
            config,
            gpu_monitor_dict
        )
            
        # Save experiment results to experiment_progress_xlsx, named as experiment_id
        if result_ws is not None:
            # Reload workbook each time before saving to avoid data loss
            with FileLock(lock_file):
                wb = openpyxl.load_workbook(experiment_progress_xlsx_path)
                # Re-fetch experiment_setting_sheet to ensure using newly loaded workbook
                experiment_setting_sheet = wb[experiment_progress_xlsx_sheet_rename_map['experiment_setting_sheet']]
                
                # Check if a sheet with the same name exists, delete old sheet if present
                if f"{experiment_id}" in wb.sheetnames:
                    del wb[f"{experiment_id}"]
                
                # Create new sheet and copy experiment results into it
                new_ws = wb.create_sheet(title=f"{experiment_id}")
                
                # Copy content from result_ws to new sheet
                for row in result_ws.iter_rows():
                    new_row = []
                    for cell in row:
                        new_row.append(cell.value)
                    new_ws.append(new_row)
                
                # Change experiment number from "ExperimentID-Batch" to "ExperimentID-0" to indicate completion
                cell = experiment_setting_sheet.cell(row=id_row, column=id_col)
                cell_value = str(cell.value).strip()
                
                # Parse experiment number format: supports "ExperimentID-Batch" or pure numbers
                match = re.match(r'^(\d+)(?:-(\d+))?$', cell_value)
                if match:
                    experiment_id_num = match.group(1)
                    # Update to "ExperimentID-0" format
                    cell.value = f"{experiment_id_num}-0"
                    print(f"Experiment {experiment_id_num} completed, cell value updated to: {cell.value}")
                else:
                    print(f"Warning: Unable to parse experiment number format: {cell_value}")
                
                wb.save(experiment_progress_xlsx_path)
            
            print(f"Experiment {experiment_id} processed and results saved to {experiment_progress_xlsx_path}")
            print(f"Experiment {experiment_id} FINISHED on GPU {gpu_id}")
                
        else:
            raise RuntimeError(f"Experiment {experiment_id} did not complete successfully")
    
    except Exception as e:
        print("Child process encountered an error:")
        traceback.print_exc()  # Print full stack trace
        raise
        
    finally:
        # Release GPU regardless of success or failure
        with lock:
            if gpu_id in gpu_usage_count:
                gpu_usage_count[gpu_id] -= 1  # Decrease GPU usage count
        print(f"GPU {gpu_id} released by experiment {experiment_id}")
    
    return pid  # Return current process PID