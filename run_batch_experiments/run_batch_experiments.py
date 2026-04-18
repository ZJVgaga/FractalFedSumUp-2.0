"""
Batch Experiment Runner Main Program
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

# Import from new modules
from gpu_monitor import GPUMonitor
from experiment_runner import run_experiment_on_gpu


def callback(pid):
    """Callback function after task completion, only logs information, does not attempt to kill the process"""
    print(f"[Callback] Task completed, PID: {pid} has ended")
    # Note: No longer attempting to kill the process, multiprocessing.Pool automatically manages process lifecycle
    # Attempting to kill an already ended process can lead to zombie processes


def error_callback(e):
    print(f"Task encountered an error: {e}")


# GPU monitor command handling thread
def gpu_monitor_command_handler(gpu_monitor, command_queue, result_dict, stop_event):
    """Handles GPU monitor commands from child processes"""
    import threading
    import time
    
    print("[GPU Monitor Command Handler] Thread started")
    
    while not stop_event.is_set():
        try:
            # Non-blocking command retrieval
            try:
                command = command_queue.get_nowait()
                if command:
                    cmd_type, pid, caller_pid = command
                    
                    if cmd_type == 'start':
                        success = gpu_monitor.start(pid)
                        result_key = f"result_{caller_pid}_{pid}_start"
                        result_dict[result_key] = ('start', pid, success, caller_pid)
                        print(f"[GPU Monitor Command Handler] Processed start({pid}) from PID {caller_pid}: {success}")
                    
                    elif cmd_type == 'end':
                        result = gpu_monitor.end(pid)
                        result_key = f"result_{caller_pid}_{pid}_end"
                        result_dict[result_key] = ('end', pid, result, caller_pid)
                        print(f"[GPU Monitor Command Handler] Processed end({pid}) from PID {caller_pid}: returned {result}")
                    
                    elif cmd_type == 'status':
                        status = {
                            'current_processes': dict(gpu_monitor.current_processes),
                            'accumulating_pids': dict(gpu_monitor.accumulating_pids),
                            'running': gpu_monitor.running
                        }
                        result_key = f"result_{caller_pid}_{pid}_status"
                        result_dict[result_key] = ('status', pid, status, caller_pid)
            except:
                pass  # Queue is empty
            
            time.sleep(0.1)  # Brief sleep to avoid high CPU usage
            
        except Exception as e:
            print(f"[GPU Monitor Command Handler] Exception: {e}")
    
    print("[GPU Monitor Command Handler] Thread stopped")

# Batch experiment process function    
def run_batch_experiments(experiment_progress_xlsx_path, hyperparam_to_cell_address_map, experiment_region, config_map):
    global config
    global experiment_progress_xlsx_sheet_rename_map
    
    # Open the existing Excel file and get the batch experiment settings sheet
    experiment_progress_xlsx = openpyxl.load_workbook(experiment_progress_xlsx_path)
    
    # Use config.get(str) to read keys from config_map in config, store results in the cell positions corresponding to the keys in config_map in the sheet named "Result Display Table_Experiment Configuration"
    # Select the corresponding sheet based on the passed sheet name
    config_sheet = experiment_progress_xlsx["Result Display Table_Experiment Configuration"]  # Use the passed experiment_region as the sheet name
    
    # Update values in the sheet
    for key, cell_address in config_map.items():
        value_to_write = config.get(key)  # Get the corresponding value from config
        config_sheet[cell_address] = value_to_write  # Write the value to the specified cell address
    
    # Save the modified Excel file
    experiment_progress_xlsx.save(experiment_progress_xlsx_path)
    
    # Categorize each experiment cell into its belonging batch according to its number, each batch stores a list,
    # each element of the list has the format: experiment id, row of the cell storing this experiment id, and column
    batch_to_experiment_id_map = get_batch_to_experiment_id_map(experiment_progress_xlsx, experiment_region)
    
    """
    batch_to_experiment_id_map format:
    {
        1: [(experiment1, row1, col1), (experiment2, row2, col2)],
        2: [(experiment3, row3, col3)],
        ...
    }
    """
    
    # First delete old erroneous experiments
    for batch, experiments in batch_to_experiment_id_map.items():
        if experiments:
            print(f"Processing experiments from {batch}th batch:")
            for experiment_id, id_row, id_col in experiments:
                # If a sheet with the same name exists, delete it
                if f"{experiment_id}" in experiment_progress_xlsx.sheetnames:
                    del experiment_progress_xlsx[f"{experiment_id}"]
    
    experiment_progress_xlsx.save(experiment_progress_xlsx_path)
    
    # Open the existing Excel file and get the batch experiment settings sheet, and unmerge all merged cells,
    # the content of merged cells will be copied to each sub-cell
    experiment_progress_xlsx = openpyxl.load_workbook(experiment_progress_xlsx_path)
    experiment_setting_sheet = experiment_progress_xlsx[experiment_progress_xlsx_sheet_rename_map['experiment_setting_sheet']]
    experiment_setting_sheet = unmerge_sheet(experiment_setting_sheet)
    
    # Save the unmerged workbook to ensure subsequent processes can obtain correct data
    experiment_progress_xlsx.save(experiment_progress_xlsx_path)
    
    # Process experiments by batch, prepare all experiment tasks
    all_experiments = []
    for batch, experiments in batch_to_experiment_id_map.items():
        if experiments:
            for experiment_id, id_row, id_col in experiments:
                all_experiments.append((experiment_id, id_row, id_col))

    # Use Manager to create shared resources
    with Manager() as manager:
        # Create GPU usage count shared dictionary
        gpu_usage_count = manager.dict()  # Record usage count for each GPU
        lock = manager.Lock()  # Create lock for managing GPU status
        
        # Create GPU monitor communication mechanism
        command_queue = manager.Queue()  # For child processes to send commands
        result_dict = manager.dict()     # For returning results
        stop_event = manager.Event()     # For stopping the command handling thread
        
        # Create GPU monitor instance
        print("[Main Process] Creating GPU monitor...")
        gpu_monitor = GPUMonitor(update_interval=1.0, enable_multiprocess=False)
        
        # Start GPU monitor
        if gpu_monitor.start_monitoring():
            print("[Main Process] GPU monitor started")
        else:
            print("[Main Process] GPU monitor failed to start")
        
        # Start command handling thread
        import threading
        command_thread = threading.Thread(
            target=gpu_monitor_command_handler,
            args=(gpu_monitor, command_queue, result_dict, stop_event),
            daemon=True
        )
        command_thread.start()
        print("[Main Process] GPU monitor command handling thread started")
        
        # Add monitor reference to config (via global variable or parameter passing)
        # Here we pass the monitor reference by modifying experiment task parameters
        # Package experiment tasks as (experiment_id, id_row, id_col, gpu_usage_count, lock, ..., gpu_monitor_dict)
        
        # Add debug information
        print(f"[Main Process] command_queue type: {type(command_queue)}, is empty: {command_queue.empty() if hasattr(command_queue, 'empty') else 'N/A'}")
        print(f"[Main Process] result_dict type: {type(result_dict)}, content: {dict(result_dict)}")
        
        gpu_tasks = [
            (str(exp[0]), int(exp[1]), int(exp[2]), gpu_usage_count, lock, 
             str(experiment_progress_xlsx_path), dict(hyperparam_to_cell_address_map),
             command_queue, result_dict)  # Add GPU monitor communication mechanism
            for exp in all_experiments
        ]

        # Check if there are experiment tasks to run
        if not gpu_tasks:
            print("[Main Process] Warning: No experiment tasks to run, possibly no experiments in Excel file or all experiments completed")
            # Stop command handling thread
            stop_event.set()
            command_thread.join(timeout=5.0)
            print("[Main Process] GPU monitor command handling thread stopped")
            
            # Stop GPU monitor
            if gpu_monitor.stop_monitoring():
                print("[Main Process] GPU monitor stopped")
            else:
                print("[Main Process] GPU monitor failed to stop")
            return  # Return directly, do not create process pool

        try:
            # Use multiprocessing pool to run experiments
            with Pool(processes=len(gpu_tasks)) as pool:
                for task in gpu_tasks:
                    pool.apply_async(
                        func=run_experiment_on_gpu, 
                        args=(task,),
                        callback=callback,       # Called when task completes 
                        error_callback=error_callback,  # Called on error 
                    )
                # Wait for all tasks to complete (otherwise main process may exit early)
                pool.close()
                pool.join()
                
        finally:
            # Stop command handling thread
            stop_event.set()
            command_thread.join(timeout=5.0)
            print("[Main Process] GPU monitor command handling thread stopped")
            
            # Stop GPU monitor
            if gpu_monitor.stop_monitoring():
                print("[Main Process] GPU monitor stopped")
            else:
                print("[Main Process] GPU monitor failed to stop")


# Configuration mapping
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


# Package all experiment configurations as a list of dictionaries, supporting multiple experiment configurations
experiments = [

    {
        "hyperparam_to_cell_address_map": {
            "Federated_Learning_Config": "A2-A13",  # Method column: FedAvg, FedProx, FedAdam, etc.
            #"dp_mechanism": "B2-B2",  # Changed from batch_size to epsilon
            #"dp_epsilon": "B3-B3",  # Changed from batch_size to epsilon
            "compressed_image_size":  "B2-B13",
            #"images_per_class":
            "Dataset": "C1-E1",  # Dataset column: MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "C2-E13",
        "hyperparameter_experiment":True,
        "experiment_xlsx_path": 'FedSumUp-hyperparameter-size.xlsx'
    },
    # FedSumUp-baselines.xlsx - Baseline experiments (updated to epsilon and MNIST/FashionMNIST/CIFAR10)
    {
        "hyperparam_to_cell_address_map": {
            "Federated_Learning_Config": "A2-A6",  # Method column: FedAvg, FedProx, FedAdam, etc.
            "dp_mechanism": "B2-B2",  # Changed from batch_size to epsilon
            #"dp_epsilon": "B3-B3",  # Changed from batch_size to epsilon
            "compressed_image_size": "B6-B6",
            "images_per_class": "B3-B5",
            "Dataset": "C1-E1",  # Dataset column: MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "C2-E6",
        "experiment_xlsx_path": 'FedSumUp-baselines-new.xlsx'
    },
{
        "hyperparam_to_cell_address_map": {
            "Federated_Learning_Config": "A2-A6",  # Method column: FedAvg, FedProx, FedAdam, etc.
            "dp_epsilon": "B2-B6",  # Changed from batch_size to epsilon
            #"dp_epsilon": "B3-B3",  # Changed from batch_size to epsilon
            #"compressed_image_size": "B6-B6",
            #"images_per_class": "B3-B5",
            "Dataset": "C1-E1",  # Dataset column: MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "C2-E6",
        "experiment_xlsx_path": 'FedSumUp-FedAvg-DP.xlsx'
    },
    {
        "hyperparam_to_cell_address_map": {
            #"Federated_Learning_Config": "A2-A6",  # Method column: FedAvg, FedProx, FedAdam, etc.
            #"dp_epsilon": "B2-B6",  # Changed from batch_size to epsilon
            #"dp_epsilon": "B3-B3",  # Changed from batch_size to epsilon
            #"compressed_image_size": "B6-B6",
            #"images_per_class": "B3-B5",
            "alpha":"B1-E1",
            "Federated_Learning_Config": "A2-A4",
            
            #"Dataset": "C1-E1",  # Dataset column: MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "B2-E4",
        "experiment_xlsx_path": 'FedSumUp-NonIID.xlsx'
    },
        {
        "hyperparam_to_cell_address_map": {
            #"Federated_Learning_Config": "A2-A6",  # Method column: FedAvg, FedProx, FedAdam, etc.
            #"dp_epsilon": "B2-B6",  # Changed from batch_size to epsilon
            #"dp_epsilon": "B3-B3",  # Changed from batch_size to epsilon
            #"compressed_image_size": "B6-B6",
            #"images_per_class": "B3-B5",
            "client_num":"B1-E1",
            "Federated_Learning_Config": "A2-A4",
            
            #"Dataset": "C1-E1",  # Dataset column: MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "B2-E4",
        "experiment_xlsx_path": 'FedSumUp-clientnum.xlsx'
    },


]
{
        "hyperparam_to_cell_address_map": {
            "Federated_Learning_Config": "A2-A6",  # Method column: FedAvg, FedProx, FedAdam, etc.
            "dp_epsilon": "B2-B6",  # Changed from batch_size to epsilon
            #"dp_epsilon": "B3-B3",  # Changed from batch_size to epsilon
            #"compressed_image_size": "B6-B6",
            #"images_per_class": "B3-B5",
            "Dataset": "C1-E1",  # Dataset column: MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "C2-E6",
        "experiment_xlsx_path": 'FedSumUp-FedAvg-DP.xlsx'
    },
{
        "hyperparam_to_cell_address_map": {
            "Federated_Learning_Config": "A2-A13",  # Method column: FedAvg, FedProx, FedAdam, etc.
            #"dp_mechanism": "B2-B2",  # Changed from batch_size to epsilon
            #"dp_epsilon": "B3-B3",  # Changed from batch_size to epsilon
            "compressed_image_size":  "B2-B13",
            #"images_per_class":
            "Dataset": "C1-E1",  # Dataset column: MNIST, FashionMNIST, CIFAR10
        },
        "experiment_region": "C2-E13",
        
        "experiment_xlsx_path": 'FedSumUp-visualize.xlsx'
    },


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    # Iterate through each experiment configuration and run
    for config in experiments:
        run_batch_experiments(
            config["experiment_xlsx_path"],
            config["hyperparam_to_cell_address_map"],
            config["experiment_region"],
            saved_config_to_excel_map  # Assuming this variable is defined externally
        )