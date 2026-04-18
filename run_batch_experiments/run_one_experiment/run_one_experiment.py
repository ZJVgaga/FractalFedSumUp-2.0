from .run_and_get_results.run_and_get_results import run_and_get_results
from .save_results_to_sheet.save_results_to_sheet import save_results_to_sheet
import random
import os
import logging

def setup_logger(name, log_dir="experiment_logger", level=logging.INFO):
    """Set up and return a logger object. Log files are saved in the ./experiment_logger/ directory."""
    
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Output to file (stored in the ./experiment_logger/ directory)
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

import time

def run_one_experiment(experiment_result_template_sheet, config, gpu_monitor_dict):
    """Run a single experiment and record logs."""
    logger = setup_logger(f"ex-id_{config.get('experiment_index')}_{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
    random.seed(config.get("seed"))
    all_results = []
    # Repeat the experiment N times
    for _ in range(config.get("experiment_times")):
        # Get experiment results
        config.set("seed", random.randint(0, 522222220))
        logger.info(f"Seed set to: {config.get('seed')}")
        result = run_and_get_results(config, logger, gpu_monitor_dict)  # Assume this function returns a dictionary in the format {result1: value1, result2: value2, ...}
        if not result:
            logger.error("Error: No experiment results obtained.")
            continue
        all_results.append(result)
    if not all_results:
        logger.error("Error: No valid experiment results.")
        return None
    # Final result dictionary
    final_result = {}
    # Get all keys
    all_keys = set().union(*[result.keys() for result in all_results])
    # Iterate over all keys
    for key in all_keys:
        processed_value = process_result_values(all_results, key)
        if processed_value is not None:
            final_result[key] = processed_value
        else:
            logger.warning(f"Key '{key}' processed result is None, skipping")

    # Print result_to_cell_map to log
    logger.info("Experiment results after multiple trials:")
    for key, value in final_result.items():
        logger.info(f"  {key}: {value}")
    
    # Create mapping from results to cell addresses
    result_to_cell_map = {
        "acc": 'B5',  # Final accuracy
        "total_communication_size": "D5",  # Total communication size
        "total_time_cost": 'E5',  # Total time cost
        "client_train_time": 'F5',
        "server_train_time": 'G5',
        "round_accuracies": "B8-B108",  # Accuracy per round
        "round_mia_accuracies": "C8-C108",  # MIA accuracy per round (new)
        "mia_acc": "C5",
        "client_gpu_sm_seconds": 'H5',  # Cumulative GPU SM utilization
        "client_cpu_time": 'I5',  # Cumulative CPU time
        # Compute budget related fields
        "compute_budget_enabled": 'J5',  # Whether compute budget is enabled
        "compute_budget_per_client_per_round": 'K5',  # Compute budget per client per round
        "budget_check_frequency": 'L5',  # Budget check frequency
        "total_flops_giga": 'M5',  # Total floating-point operations (in billions)
        "avg_flops_per_client_per_round_giga": 'N5',  # Average floating-point operations per client per round
        "budget_utilization_percent": 'O5',  # Budget utilization percentage
        "clients_exceeded_budget": 'P5',  # Number of clients exceeding budget
        "avg_epochs_per_client": 'Q5',  # Average training epochs per client
    }
    try:
        clone_sheet = save_results_to_sheet(config, experiment_result_template_sheet, final_result, result_to_cell_map)
        logger.info("Experiment results saved successfully.")
        return clone_sheet
    except Exception as e:
        logger.error(f"Failed to save results: {e}", exc_info=True)
        return None

import numpy as np

def calculate_mean_std(values):
    """
    Calculate the mean and standard deviation, and return as a string in the format "A±B".
    Parameters:
    - values: A list or array containing numerical values.
    Returns:
    - String representation of mean and standard deviation in the format "A±B".
    """
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)  # Use sample standard deviation
    if std_val is None:
        return f"{mean_val:.4f}"
    return f"{mean_val:.4f}±{std_val:.4f}"

def process_result_values(results_list, key):
    """
    Calculate mean and standard deviation based on the type of result values (list, number, dictionary, or other).
    Parameters:
    - results_list: A list containing results from multiple experiments, each result is a dictionary.
    - key: The key currently being processed.
    Returns:
    - Processed result value in the format "A±B" or the original value (if mean and standard deviation cannot be calculated).
    """
    # Check if the key contains a dot (indicating a nested dictionary)
    if '.' in key:
        # Process nested keys, e.g., "flops_stats.total_flops_giga"
        parts = key.split('.')
        
        # Extract values for this nested key from all results
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
            # If no result contains this path, return None
            return None
        
        # Get the first valid value for type checking
        first_value = values[0]
        
        # Check the type of the value
        if isinstance(first_value, list):
            # If it's a list, calculate mean and standard deviation for each corresponding element position
            transposed_lists = zip(*values)
            processed_values = [calculate_mean_std(list(vals)) for vals in transposed_lists]
            return processed_values
        elif isinstance(first_value, (int, float)):
            # If it's a number, calculate mean and standard deviation
            return calculate_mean_std(values)
        else:
            # For other data types, take the value from the first result
            return first_value
    else:
        # Process regular keys
        if key not in results_list[0]:
            return None
            
        first_value = results_list[0][key]
        if isinstance(first_value, list):
            # If it's a list, calculate mean and standard deviation for each corresponding element position
            transposed_lists = zip(*[result[key] for result in results_list])
            processed_values = [calculate_mean_std(list(vals)) for vals in transposed_lists]
            return processed_values
        elif isinstance(first_value, (int, float)):
            # If it's a number, calculate mean and standard deviation
            values = [result[key] for result in results_list]
            return calculate_mean_std(values)
        else:
            # For other data types, take the value from the first result
            return first_value