import os
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Color
from openpyxl.drawing.image import Image as XLImage
from openpyxl import load_workbook

# Define a list containing multiple Excel file paths
excel_file_paths = [

    #"FedSumUp-hyperparameter-size.xlsx",
    "FedSumUp-FedAvg-DP.xlsx"
] 
image_dir = "experiment_image_result"


def find_cells_by_color(sheet, target_fill):
    """Find all cells with the specified color"""
    matching_cells = []
    for row in sheet.iter_rows():
        for cell in row:
            if hasattr(cell.fill, 'start_color') and cell.fill.start_color.rgb == target_fill.start_color.rgb:
                matching_cells.append(cell)
    return matching_cells

def get_fill_from_cell(cell):
    """Extract fill information from a cell"""
    start_color_rgb = cell.fill.start_color.rgb if hasattr(cell.fill, 'start_color') and isinstance(cell.fill.start_color.rgb, str) else None
    if start_color_rgb:
        start_color = Color(rgb=start_color_rgb)
    else:
        start_color = None
    
    return PatternFill(
        start_color=start_color,
        fill_type='solid'
    )

def update_cell_with_template_data(workbook, result_sheet, cell, template_sheet_name, image_dir=image_dir):
    """
    Find the corresponding cell position in the template worksheet based on the cell's color,
    and update the target cell with data from that position.
    Simultaneously insert the experiment result image stored in the same path into the table.
    """
    # Get the cell's color
    cell_fill = get_fill_from_cell(cell)

    # Find cells with the same color in the template worksheet
    template_sheet = workbook[template_sheet_name]
    matching_cells_in_template = find_cells_by_color(template_sheet, cell_fill)
    
    if matching_cells_in_template:
        # Assume we only process the first matching cell
        template_cell = matching_cells_in_template[0]
        target_sheet_name = str(cell.value)  # Use the cell value as the worksheet name
        
        if target_sheet_name in workbook.sheetnames:
            target_sheet = workbook[target_sheet_name]
            target_value = target_sheet[template_cell.coordinate].value
            
            # Update the cell's value and color
            cell.value = target_value
            cell.fill = cell_fill  # Keep the original color unchanged
            
            # Attempt to insert the corresponding image
            image_filename = f"{target_value}"  # Assume the image filename is the target value plus .png extension
            image_path = os.path.join(os.path.dirname(excel_file_path), image_dir, image_filename)
            
            if os.path.exists(image_path):
                img = XLImage(image_path)
                cell_address = cell.coordinate
                result_sheet.add_image(img, cell_address)
            else:
                print(f"No image found for {target_value} at path: {image_path}")
        else:
            print(f"Sheet {target_sheet_name} not found.")
    else:
        print(f"No matching cell found in template sheet for cell {cell.coordinate}.")

def create_or_replace_result_sheet(workbook, template_sheet_name):
    """Create or replace the worksheet named 'Result Display Table_'"""
    # Generate a new worksheet name based on the template worksheet name
    if "Result Display Template" in template_sheet_name:
        new_sheet_name = template_sheet_name.replace("Result Display Template", "Result Display Table")
    else:
        raise ValueError(f"Template worksheet name should contain 'Result Display Template', actual name: {template_sheet_name}")
    
    # If a worksheet with the same name already exists in the workbook, delete it
    if new_sheet_name in workbook.sheetnames:
        del workbook[new_sheet_name]
    
    # Copy the template worksheet as a new worksheet
    template_sheet = workbook[template_sheet_name]
    new_sheet = workbook.copy_worksheet(template_sheet)
    new_sheet.title = new_sheet_name
    
    return new_sheet

def process_result_sheets(workbook, template_sheet_name):
    """Process all worksheets starting with 'Result Display Template'"""
    for sheet_name in workbook.sheetnames:
        if sheet_name.startswith("Result Display Template"):
            # Create or replace 'Result Display Table_'
            result_sheet = create_or_replace_result_sheet(workbook,sheet_name)
            
            # Iterate through each cell in the current worksheet
            for row in result_sheet.iter_rows():
                for cell in row:
                    # Check if the cell has a fill color and it's not the default color
                    if hasattr(cell.fill, 'start_color') and cell.fill.start_color.index not in ['00000000', 'FFFFFFFF', None]:
                        update_cell_with_template_data(workbook, result_sheet, cell, template_sheet_name)
# Load the workbook


for excel_file_path in excel_file_paths:
    try:
        # Load the workbook
        workbook = load_workbook(filename=excel_file_path)

        # Call the function to process all eligible worksheets
        process_result_sheets(workbook, "Experiment Result Record Template")

        # Save the workbook
        workbook.save(excel_file_path)
        print(f"Successfully processed and saved file: {excel_file_path}")
    except Exception as e:
        print(f"Error occurred while processing file {excel_file_path}: {e}")