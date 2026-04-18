import os
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Color
from openpyxl.drawing.image import Image as XLImage
from openpyxl import load_workbook

# 定义包含多个 Excel 文件路径的列表
excel_file_paths = [

    #"FedSumUp-hyperparameter-size.xlsx",
    "FedSumUp-FedAvg-DP.xlsx"
] 
image_dir = "experiment_image_result"


def find_cells_by_color(sheet, target_fill):
    """查找指定颜色的所有单元格"""
    matching_cells = []
    for row in sheet.iter_rows():
        for cell in row:
            if hasattr(cell.fill, 'start_color') and cell.fill.start_color.rgb == target_fill.start_color.rgb:
                matching_cells.append(cell)
    return matching_cells

def get_fill_from_cell(cell):
    """从单元格中提取填充信息"""
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
    根据单元格的颜色在模板工作表中找到对应单元格的位置，
    并使用相应位置的数据更新目标单元格。
    同时将同路径下存放的实验结果img插入到表格当中。
    """
    # 获取单元格的颜色
    cell_fill = get_fill_from_cell(cell)

    # 在模板工作表中查找具有相同颜色的单元格
    template_sheet = workbook[template_sheet_name]
    matching_cells_in_template = find_cells_by_color(template_sheet, cell_fill)
    
    if matching_cells_in_template:
        # 假设我们只处理第一个匹配的单元格
        template_cell = matching_cells_in_template[0]
        target_sheet_name = str(cell.value)  # 使用单元格值作为工作表名
        
        if target_sheet_name in workbook.sheetnames:
            target_sheet = workbook[target_sheet_name]
            target_value = target_sheet[template_cell.coordinate].value
            
            # 更新单元格的值和颜色
            cell.value = target_value
            cell.fill = cell_fill  # 保持原有颜色不变
            
            # 尝试插入对应的图片
            image_filename = f"{target_value}"  # 假定图片文件名为目标值加上.png后缀
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
    """创建或替换名为“结果展示表_”的工作表"""
    # 根据模板工作表名生成新的工作表名
    if "结果展示模板" in template_sheet_name:
        new_sheet_name = template_sheet_name.replace("结果展示模板", "结果展示表")
    else:
        raise ValueError(f"模板工作表名称应包含'结果展示模板', 实际名称: {template_sheet_name}")
    
    # 如果工作簿中已存在同名的工作表，则删除它
    if new_sheet_name in workbook.sheetnames:
        del workbook[new_sheet_name]
    
    # 复制模板工作表作为新工作表
    template_sheet = workbook[template_sheet_name]
    new_sheet = workbook.copy_worksheet(template_sheet)
    new_sheet.title = new_sheet_name
    
    return new_sheet

def process_result_sheets(workbook, template_sheet_name):
    """处理所有以“结果展示模板”开头的工作表"""
    for sheet_name in workbook.sheetnames:
        if sheet_name.startswith("结果展示模板"):
            # 创建或替换“结果展示表_”
            result_sheet = create_or_replace_result_sheet(workbook,sheet_name)
            
            # 遍历当前工作表中的每个单元格
            for row in result_sheet.iter_rows():
                for cell in row:
                    # 检查单元格是否有填充颜色且不是默认颜色
                    if hasattr(cell.fill, 'start_color') and cell.fill.start_color.index not in ['00000000', 'FFFFFFFF', None]:
                        update_cell_with_template_data(workbook, result_sheet, cell, template_sheet_name)
# 加载工作簿


for excel_file_path in excel_file_paths:
    try:
        # 加载工作簿
        workbook = load_workbook(filename=excel_file_path)

        # 调用函数处理所有符合条件的工作表
        process_result_sheets(workbook, "实验结果记录模板")

        # 保存工作簿
        workbook.save(excel_file_path)
        print(f"成功处理并保存文件: {excel_file_path}")
    except Exception as e:
        print(f"处理文件 {excel_file_path} 时出现错误: {e}")