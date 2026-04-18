import openpyxl
from openpyxl.utils.cell import get_column_letter

# Define a helper function to search for experiment parameters in a specified direction from a given experiment cell,
# and set those parameters into the experiment configuration Config.
# For a cell storing an experiment number, there must be cells above or to the left that store the experiment parameters
# this experiment intends to modify.
# We collect all the experiment parameters that this experiment cell wants to change.
# Then, we update the specific values of these parameters into the corresponding experiment's Config.
# The values of experiment parameters in Config that are not changed use the default values from Config initialization.
# This ensures that in different experiments, such as Experiment A and Experiment B, only the values of the modified
# experiment parameters differ, while the values of unmodified parameters remain the same, all using Config's defaults.
# This achieves the goal of controlling experimental variables and ensures the reliability of experimental results.
def find_and_set_config(config, experiment_setting_sheet, cell_address_to_hyperparam_map, direction, start_row, start_col):
        row, col = start_row, start_col  # Initialize the position of an experiment number
        while True:
            # Move row or column based on direction
            if direction == 'up':  # Move upward
                row -= 1
            elif direction == 'left':  # Move leftward
                col -= 1
            
            # Boundary check to prevent out-of-range
            if row < 1 or col < 1:
                break  # Stop searching if beyond boundary
            
            # Construct the address of the current cell (e.g., "A1")
            cell_address = f'{get_column_letter(col)}{row}'
            
            # Check if the current cell address exists in cell_address_to_hyperparam_map
            if cell_address in cell_address_to_hyperparam_map:
                config_key = cell_address_to_hyperparam_map[cell_address]  # Get the configuration item name
                cell_value = experiment_setting_sheet[cell_address].value  # Get the cell's value
                
                # If the cell's value is not None, it means this cell stores the variable value the experiment needs to modify,
                # and we update its value into the configuration.
                if cell_value is not None:
                    # Handle string values: strip leading and trailing spaces
                    if isinstance(cell_value, str):
                        cell_value = cell_value.strip()
                        # Attempt to convert the string to an appropriate type
                        try:
                            # Check for boolean values first (since 'true' contains 'e')
                            if cell_value.lower() in ['true', 'false']:
                                cell_value = cell_value.lower() == 'true'
                            # Check for integers
                            elif cell_value.isdigit() or (cell_value.startswith('-') and cell_value[1:].isdigit()):
                                cell_value = int(cell_value)
                            # Check for floats
                            elif '.' in cell_value:
                                # Verify it's actually a float format (not some other string containing a dot)
                                parts = cell_value.split('.')
                                if len(parts) == 2 and parts[0].lstrip('-').isdigit() and parts[1].isdigit():
                                    cell_value = float(cell_value)
                            # Check for scientific notation
                            elif 'e' in cell_value.lower():
                                # Verify scientific notation format
                                parts = cell_value.lower().split('e')
                                if len(parts) == 2 and (parts[0].replace('.', '').replace('-', '').isdigit() or 
                                                       (parts[0].startswith('-') and parts[0][1:].replace('.', '').isdigit())) and \
                                   parts[1].lstrip('-').isdigit():
                                    cell_value = float(cell_value)
                        except (ValueError, AttributeError):
                            # Conversion failed, keep the original string
                            pass
                    
                    config.set(config_key, cell_value)  # Update the configuration tree
                    print(f"Set {config_key} to {cell_value}")  # Print log

        return config