import openpyxl
from openpyxl.utils.cell import column_index_from_string, get_column_letter


# Define a function to convert a hyperparam to cell address mapping into a cell address to hyperparam mapping
def reverse_hyperparam_to_cell_address_map(hyperparam_to_cell_address_map):
    """
    Convert hyperparam_to_cell_address_map into cell_address_to_hyperparam_map

    Input format:
        hyperparam_to_cell_address_map: dict
            {
                "config_key_1": "start_cell-end_cell",  # Single region
                "config_key_2": ["start_cell1-end_cell1", "start_cell2-end_cell2"],  # Multiple regions
                ...
            }

    Output format:
        cell_address_to_hyperparam_map: dict
            {"A1": "config_key_1", "A2": "config_key_1", ...}
    """
    # Initialize an empty dictionary to store the mapping from cell to configuration item
    
    cell_address_to_hyperparam_map = {}
    
    # Iterate through each configuration item and its corresponding cell range in hyperparam_to_cell_address_map
    for config_key, cell_ranges in hyperparam_to_cell_address_map.items():
        # If the cell range is a list (indicating multiple regions)
        if isinstance(cell_ranges, list):
            # Iterate through each cell range in the list
            for cell_range in cell_ranges:
                # Split the cell range string into start cell and end cell
                start_cell, end_cell = cell_range.split('-')
                
                # Extract row and column numbers from the start cell
                row_start = int(start_cell[1:])  # Extract row number (e.g., "A1" -> 1)
                col_start = column_index_from_string(start_cell[:1])  # Extract column number (e.g., "A1" -> 1)
                
                # Extract row and column numbers from the end cell
                row_end = int(end_cell[1:])  # Extract row number (e.g., "B2" -> 2)
                col_end = column_index_from_string(end_cell[:1])  # Extract column number (e.g., "B2" -> 2)

                # Iterate through all rows and columns in the current cell range
                for row in range(row_start, row_end + 1):  # Iterate row numbers
                    for col in range(col_start, col_end + 1):  # Iterate column numbers
                        # Construct the cell address (e.g., "A1")
                        cell_address = f'{get_column_letter(col)}{row}'
                        
                        # Map the cell address to the corresponding configuration item
                        # Format: {"A1": "config_key_1", "A2": "config_key_1", ...}
                        cell_address_to_hyperparam_map[cell_address] = config_key
        else:
            # If the cell range is not a list (indicating a single region)
            # Split the cell range string into start cell and end cell
            start_cell, end_cell = cell_ranges.split('-')
            
            # Extract row and column numbers from the start cell
            row_start = int(start_cell[1:])  # Extract row number (e.g., "A1" -> 1)
            col_start = column_index_from_string(start_cell[:1])  # Extract column number (e.g., "A1" -> 1)
            
            # Extract row and column numbers from the end cell
            row_end = int(end_cell[1:])  # Extract row number (e.g., "B2" -> 2)
            col_end = column_index_from_string(end_cell[:1])  # Extract column number (e.g., "B2" -> 2)

            # Iterate through all rows and columns in the current cell range
            for row in range(row_start, row_end + 1):  # Iterate row numbers
                for col in range(col_start, col_end + 1):  # Iterate column numbers
                    # Construct the cell address (e.g., "A1")
                    cell_address = f'{get_column_letter(col)}{row}'
                    
                    # Map the cell address to the corresponding configuration item
                    # Format: {"A1": "config_key_1", "A2": "config_key_1", ...}
                    cell_address_to_hyperparam_map[cell_address] = config_key
    
    # Return the generated mapping from cell to configuration item
    # Format: {"A1": "config_key_1", "A2": "config_key_1", ...}
    return cell_address_to_hyperparam_map