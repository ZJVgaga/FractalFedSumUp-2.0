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
    创建一个现有工作表的深拷贝。
    参数:
    - sheet (Worksheet): 要复制的工作表。
    返回:
    - cloned_sheet (Worksheet): 工作表的深拷贝副本。
    """
    # 创建一个新的工作表
    wb = sheet._parent
    cloned_sheet = wb.create_sheet(title=sheet.title + "_copy")
    # 复制单元格内容和样式
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
    # 复制合并单元格
    for merged_cells in sheet.merged_cells.ranges:
        cloned_sheet.merge_cells(str(merged_cells))
    # 复制列宽
    for col_letter, column_dimension in sheet.column_dimensions.items():
        cloned_sheet.column_dimensions[col_letter] = copy(column_dimension)
    return cloned_sheet

import os
from openpyxl.drawing.image import Image as XLImage

def save_results_to_sheet(config,experiment_result_template_sheet, results, result_to_cell_map, image_save_dir="experiment_image_result"):
    """
    将实验结果保存到指定的工作表中的对应单元格中，支持文本和图片。
    参数：
    - experiment_result_template_sheet (openpyxl.worksheet.worksheet.Worksheet): 
      用于保存结果的目标工作表。
    - results (dict): 包含实验结果的字典，键为结果名称，值为结果数据（可以是文本或PIL图像）。
    - result_to_cell_map (dict): 结果名称与单元格地址的映射关系。
    - experiment_index (int or str): 实验索引，用于命名保存的图像文件。
    - image_save_dir (str): 图像保存目录，默认为 "experiment_image_result"。

    返回：
    - cloned_sheet (openpyxl.worksheet.worksheet.Worksheet): 更新后的结果工作表。
    """
    # 创建指定工作表的一个副本
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
            # 如果结果是一个 PIL 图像对象
            if isinstance(result_value, Image.Image):
                # 确定图像保存路径和文件名
                image_filename = f"{config.get('experiment_index')}-{cell_address.replace(':', '_')}.png"
                image_path = os.path.join(image_save_dir, image_filename)
                #如果存在同名的图像文件，则删除它
                if os.path.exists(image_path):
                    os.remove(image_path)
                # 保存图像
                result_value.save(image_path)
                # 插入图片到 Excel 单元格
                img = XLImage(image_path)
                img_anchor = cell_address
                cloned_sheet.add_image(img, img_anchor)
                # 更新单元格内容为图片的文件名
                cloned_sheet[cell_address].value = image_filename
            elif isinstance(result_value, list):
                # 如果结果是列表，则根据提供的地址范围填充数据
                if '-' in cell_address:  # 检查是否是一个范围
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