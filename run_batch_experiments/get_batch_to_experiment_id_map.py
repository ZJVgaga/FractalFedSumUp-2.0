from xlsx_tools import *
import re

# The experiment number format is "experiment ID-batch", e.g., "1-1" means experiment 1 in batch 1
def get_batch_to_experiment_id_map(experiment_progress_xlsx, experiment_region):
    """
    Generates a mapping of experiment IDs stored by batch based on the data in the workbook.

    Parameters:
        experiment_progress_xlsx (openpyxl.experiment_progress_xlsx): The input workbook object.
        experiment_region (str): The experiment area range, formatted as "A1:B10".

    Returns:
        dict: A dictionary storing experiment IDs by batch, where keys are batch numbers (integers) and values are lists containing experiment information.
              Example output format:
              {
                  1: [("Experiment1", row1, col1), ("Experiment2", row2, col2)],
                  2: [("Experiment3", row3, col3)]
              }
    """
    global experiment_progress_xlsx_sheet_rename_map
    # No longer using a color mapping table, directly parsing batch from experiment number

    # Get the batch experiment settings sheet
    try:
        experiment_setting_sheet = experiment_progress_xlsx[experiment_progress_xlsx_sheet_rename_map["experiment_setting_sheet"]]
    except KeyError:
        print(f"Error: Worksheet {experiment_progress_xlsx_sheet_rename_map['experiment_setting_sheet']} does not exist in the workbook.")
        return {}

    # Parse the experiment region to determine its start and end row and column numbers
    start_cell, end_cell = experiment_region.split('-')
    (row_start, col_start) = parse_cell(start_cell)
    (row_end, col_end) = parse_cell(end_cell)

    # Ensure the row and column order is correct (top-left to bottom-right)
    if row_start > row_end:
        row_start, row_end = row_end, row_start
    if col_start > col_end:
        col_start, col_end = col_end, col_start

    # Generate the batch-to-experiment ID mapping, initially empty
    batch_to_experiment_id_map = {}

    # Traverse all cells within the experiment region
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            cell_value = experiment_setting_sheet.cell(row=row, column=col).value
            if cell_value is not None:
                # Parse experiment number and batch
                # Format should be: "experiment ID-batch" or a plain number (defaults to batch 1)
                cell_str = str(cell_value).strip()
                
                # Attempt to parse the format "number-number"
                match = re.match(r'^(\d+)(?:-(\d+))?$', cell_str)
                
                if match:
                    experiment_id = match.group(1)
                    batch_str = match.group(2)
                    
                    if batch_str:
                        # Has batch information, e.g., "1-1"
                        batch_number = int(batch_str)
                        if batch_number > 0:
                            batch_name = f"Batch {batch_number}"
                            print(f"Setting experiment {experiment_id} as {batch_name}.")
                            batch_to_experiment_id_map.setdefault(batch_number, []).append((experiment_id, row, col))
                        elif batch_number == 0:
                            print(f"Skipping experiment {experiment_id} as it's marked as 'already completed'.")
                        elif batch_number == -1:
                            print(f"Skipping experiment {experiment_id} as it's marked as 'not required'.")
                    else:
                        # No batch information, defaults to batch 1, e.g., "1"
                        batch_number = 1
                        batch_name = f"Batch {batch_number}"
                        print(f"Setting experiment {experiment_id} as {batch_name} (default).")
                        batch_to_experiment_id_map.setdefault(batch_number, []).append((experiment_id, row, col))
                else:
                    print(f"Cannot parse experiment number format: {cell_str}")
    
    # Sort the dictionary mapping by batch order, i.e., sort the dictionary by keys in ascending order. The sorted dictionary format is:
    '''
    {-1:[3,1,5],# Indicates experiments numbered 3, 1, and 5 are not required
      0:[4,2,6],# Indicates experiments numbered 4, 2, and 6 are already completed
      1:[9],# Indicates experiment numbered 9 is in batch 1
      2:[10],# Indicates experiment numbered 10 is in batch 2
      ...
    }
    '''
    sorted_batch_to_experiment_id_map = {key: batch_to_experiment_id_map[key] for key in sorted(batch_to_experiment_id_map)}
    return sorted_batch_to_experiment_id_map