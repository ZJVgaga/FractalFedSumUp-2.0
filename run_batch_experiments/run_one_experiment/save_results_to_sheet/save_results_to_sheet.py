import openpyxl
from openpyxl.drawing.image import Image as XLImage
from PIL import Image
import os
from copy import copy
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.utils import get_column_letter, column_index_from_string

def clone_sheet(sheet: Worksheet) -> Worksheet:
    """
    Create a deep copy of an existing worksheet.
    Parameters:
    - sheet (Worksheet): The worksheet to be copied.
    Returns:
    - cloned_sheet (Worksheet): A deep copy of the worksheet.
    """
    # Create a new worksheet
    wb = sheet._parent
    cloned_sheet = wb.create_sheet(title=sheet.title + "_copy")
    # Copy cell content and styles
    for row in sheet.iter_rows():
        new_row = cloned_sheet.row_dimensions[row[0].row]
        new_row.height = copy(sheet.row_dimensions[row[0].row].height)
        for cell in row:
            new_cell = cloned_sheet.cell(row=cell.row, column=cell.column, value=copy(cell.value))
            if cell.has_style:
                new_cell.font = copy(cell.font)
                new_cell.border = copy(cell.border)
                new_cell.fill = copy(cell.fill)
                new_cell.number_format = copy(cell.number_format)
                new_cell.protection = copy(cell.protection)
                new_cell.alignment = copy(cell.alignment)
    # Copy merged cells
    for merged_cells in sheet.merged_cells.ranges:
        cloned_sheet.merge_cells(str(merged_cells))
    # Copy column widths
    for col_letter, column_dimension in sheet.column_dimensions.items():
        cloned_sheet.column_dimensions[col_letter] = copy(column_dimension)
    return cloned_sheet

import os
from openpyxl.drawing.image import Image as XLImage

def save_results_to_sheet(config, experiment_result_template_sheet, results, result_to_cell_map, image_save_dir="experiment_image_result"):
    """
    Save experiment results to corresponding cells in the specified worksheet, supporting both text and images.
    Parameters:
    - experiment_result_template_sheet (openpyxl.worksheet.worksheet.Worksheet):
      The target worksheet for saving results.
    - results (dict): A dictionary containing experiment results, where keys are result names and values are result data (can be text or PIL images).
    - result_to_cell_map (dict): A mapping of result names to cell addresses.
    - experiment_index (int or str): Experiment index used for naming saved image files.
    - image_save_dir (str): Directory for saving images, defaults to "experiment_image_result".

    Returns:
    - cloned_sheet (openpyxl.worksheet.worksheet.Worksheet): The updated result worksheet.
    """
    # Create a copy of the specified worksheet
    cloned_sheet = clone_sheet(experiment_result_template_sheet)
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    def range_to_cells(start_cell, end_cell):
        start_col, start_row = start_cell[0], int(start_cell[1:])
        end_col, end_row = end_cell[0], int(end_cell[1:])
        start_col_num = ord(start_col) - 64
        end_col_num = ord(end_col) - 64
        cells = []
        for row in range(start_row, end_row + 1):
            for col in range(start_col_num, end_col_num + 1):
                cells.append(f"{get_column_letter(col)}{row}")
        return cells
    for result_key, cell_address in result_to_cell_map.items():
        if result_key in results:
            result_value = results[result_key]
            # If the result is a PIL Image object
            if isinstance(result_value, Image.Image):
                # Determine the image save path and filename
                image_filename = f"{config.get('experiment_index')}-{cell_address.replace(':', '_')}.png"
                image_path = os.path.join(image_save_dir, image_filename)
                # If an image file with the same name exists, delete it
                if os.path.exists(image_path):
                    os.remove(image_path)
                # Save the image
                result_value.save(image_path)
                # Insert the image into the Excel cell
                img = XLImage(image_path)
                img_anchor = cell_address
                cloned_sheet.add_image(img, img_anchor)
                # Update the cell content with the image filename
                cloned_sheet[cell_address].value = image_filename
            elif isinstance(result_value, list):
                # If the result is a list, fill data according to the provided address range
                if '-' in cell_address:  # Check if it's a range
                    start_cell, end_cell = cell_address.split('-')
                    cells = range_to_cells(start_cell, end_cell)
                    for idx, value in enumerate(result_value):
                        if idx < len(cells):
                            cloned_sheet[cells[idx]] = str(value)
                        else:
                            break
                else:
                    cloned_sheet[cell_address] = ','.join(map(str, result_value))
            else:
                cloned_sheet[cell_address] = str(result_value)
        else:
            print(f"Warning: Result key '{result_key}' not found in results dictionary.")
    return cloned_sheet