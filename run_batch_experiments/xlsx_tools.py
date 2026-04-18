import openpyxl
from openpyxl.utils.cell import column_index_from_string

# Maintain a mapping of sheet names and their meanings in the experiment progress xlsx
experiment_progress_xlsx_sheet_rename_map={
    'experiment_setting_sheet':'Batch Experiment Settings Sheet',
    'one_experiment_result_template_sheet':"Experiment Result Recording Template",
}

# Define a function to unmerge cells and copy the content to each cell
def unmerge_sheet(ws):
    """

    Input:
        ws (openpyxl.Worksheet): A worksheet object containing merged cells.

    Output:
        openpyxl.Worksheet: The worksheet object with merged cells split and content copied.
    """
    # Iterate through all merged cell ranges in the worksheet
    for merge in ws.merged_cells.ranges.copy():
        # Get the boundary information of the merged cell range
        # bounds returns four values: min column number, min row number, max column number, max row number
        min_col, min_row, max_col, max_row = merge.bounds
        
        # Get the value of the top-left cell (i.e., the first cell) in the merged range
        top_left_value = ws.cell(row=min_row, column=min_col).value
        
        # Unmerge the current merged cell range
        ws.unmerge_cells(str(merge))
        
        # Iterate through all cells within the merged cell range
        for row in range(min_row, max_row + 1):  # Iterate rows
            for col in range(min_col, max_col + 1):  # Iterate columns
                # Copy the value from the top-left cell to each cell
                ws.cell(row=row, column=col, value=top_left_value)
    # Return the modified worksheet object
    return ws




"""Parse a cell string (e.g., 'J3') into column index and row number, returns J and 3"""
def parse_cell(cell_str):
    """Parse a cell string (e.g., 'J3') into column index and row number"""
    col_str = ''.join(filter(str.isalpha, cell_str))
    row_str = ''.join(filter(str.isdigit, cell_str))
    return int(row_str), column_index_from_string(col_str)