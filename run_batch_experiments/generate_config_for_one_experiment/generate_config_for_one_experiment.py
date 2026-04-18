from .reverse_hyperparam_to_cell_address_map import reverse_hyperparam_to_cell_address_map
from .find_and_set_config import find_and_set_config
import openpyxl
from openpyxl.utils.cell import get_column_letter


# Define a function to extract experiment parameter values from the experiment settings sheet for a single experiment and update its configuration
def generate_config_for_one_experiment(experiment_setting_sheet, hyperparam_to_cell_address_map, experiment_id, current_row, current_col, config):
    
    """
    Extract configuration values from the worksheet and update the configuration tree based on the cell address mapping.

    Parameters:
        experiment_setting_sheet (openpyxl.Worksheet): The input worksheet object.
        hyperparam_to_cell_address_map (dict): A mapping dictionary from cell addresses to configuration item names.
                          Format: {"A1": "config_key_1", "B2": "config_key_2", ...}
        experiment_id (int): The index or ID of the current experiment.
        current_row (int): The row number of the current cell.
        current_col (int): The column number of the current cell.

    Returns:
        config(ConfigTree): The configuration tree for the specified experiment.
    """
    # Reverse the mapping
    cell_address_to_hyperparam_map = reverse_hyperparam_to_cell_address_map(hyperparam_to_cell_address_map)
    """
    cell_address_to_hyperparam_map format:
            {"A1": "config_key_1", "A2": "config_key_1", ...}
    """
    
    # Traverse upward to find configuration items in the cells above
    config = find_and_set_config(config, experiment_setting_sheet, cell_address_to_hyperparam_map, 'up', current_row, current_col)
    
    # Traverse leftward to find configuration items in the cells to the left
    config = find_and_set_config(config, experiment_setting_sheet, cell_address_to_hyperparam_map, 'left', current_row, current_col)
    
    # Set the current experiment's index or ID into the configuration tree
    config.set("experiment_index", experiment_id)
    return config