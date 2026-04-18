import openpyxl
from openpyxl.utils.cell import column_index_from_string

#维护一个实验过程xlsx的sheet名称和含义
experiment_progress_xlsx_sheet_rename_map={
    'experiment_setting_sheet':'批量实验设置表',
    'one_experiment_result_template_sheet':"实验结果记录模板",
}

# 定义一个函数，用于取消合并单元格并将内容复制到每一个单元格中
def unmerge_sheet(ws):
    """

    输入:
        ws (openpyxl.Worksheet): 存在合并单元格的工作表对象。

    输出:
        openpyxl.Worksheet: 拆分并复制了内容的工作表对象。
    """
    # 遍历工作表中所有已合并的单元格区域
    for merge in ws.merged_cells.ranges.copy():
        # 获取合并单元格区域的边界信息
        # bounds 返回四个值：最小列号、最小行号、最大列号、最大行号
        min_col, min_row, max_col, max_row = merge.bounds
        
        # 获取合并单元格左上角（即第一个单元格）的值
        top_left_value = ws.cell(row=min_row, column=min_col).value
        
        # 取消当前合并单元格区域的合并状态
        ws.unmerge_cells(str(merge))
        
        # 遍历合并单元格区域内的所有单元格
        for row in range(min_row, max_row + 1):  # 遍历行
            for col in range(min_col, max_col + 1):  # 遍历列
                # 将左上角单元格的值复制到每个单元格中
                ws.cell(row=row, column=col, value=top_left_value)
    # 返回修改后的工作表对象
    return ws




"""将单元格字符串（如 'J3'）解析为列索引和行号，返回J和3"""
def parse_cell(cell_str):
    """将单元格字符串（如 'J3'）解析为列索引和行号"""
    col_str = ''.join(filter(str.isalpha, cell_str))
    row_str = ''.join(filter(str.isdigit, cell_str))
    return int(row_str), column_index_from_string(col_str)
