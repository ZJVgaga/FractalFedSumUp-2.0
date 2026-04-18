from xlsx_tools import *
import re

#实验编号格式为"实验ID-批次"，例如"1-1"表示实验1在第1批
def get_batch_to_experiment_id_map(experiment_progress_xlsx, experiment_region):
    """
    根据工作簿中的数据，生成按批次存储实验 ID 的映射。

    参数:
        experiment_progress_xlsx (openpyxl.experiment_progress_xlsx): 输入的工作簿对象。
        experiment_region (str): 实验区域范围，格式为 "A1:B10"。

    返回:
        dict: 按批次存储实验 ID 的字典，键为批次编号（整数），值为包含实验信息的列表。
              示例输出格式：
              {
                  1: [("实验1", row1, col1), ("实验2", row2, col2)],
                  2: [("实验3", row3, col3)]
              }
    """
    global experiment_progress_xlsx_sheet_rename_map
    # 不再使用颜色映射表，直接从实验编号解析批次

    # 获取批量实验设置表
    try:
        experiment_setting_sheet = experiment_progress_xlsx[experiment_progress_xlsx_sheet_rename_map["experiment_setting_sheet"]]
    except KeyError:
        print(f"错误：工作簿中不存在 {experiment_progress_xlsx_sheet_rename_map['experiment_setting_sheet']} 工作表。")
        return {}

    # 解析实验区域，确定其起止行、列号
    start_cell, end_cell = experiment_region.split('-')
    (row_start, col_start) = parse_cell(start_cell)
    (row_end, col_end) = parse_cell(end_cell)

    # 确保行和列的顺序正确（左上角到右下角）
    if row_start > row_end:
        row_start, row_end = row_end, row_start
    if col_start > col_end:
        col_start, col_end = col_end, col_start

    # 生成批次-实验编号映射，初始为空
    batch_to_experiment_id_map = {}

    # 遍历实验区域内的所有单元格
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            cell_value = experiment_setting_sheet.cell(row=row, column=col).value
            if cell_value is not None:
                # 解析实验编号和批次
                # 格式应为: "实验ID-批次" 或纯数字（默认为第1批）
                cell_str = str(cell_value).strip()
                
                # 尝试解析格式 "数字-数字"
                match = re.match(r'^(\d+)(?:-(\d+))?$', cell_str)
                
                if match:
                    experiment_id = match.group(1)
                    batch_str = match.group(2)
                    
                    if batch_str:
                        # 有批次信息，如 "1-1"
                        batch_number = int(batch_str)
                        if batch_number > 0:
                            batch_name = f"第{batch_number}批"
                            print(f"Setting experiment {experiment_id} as {batch_name}.")
                            batch_to_experiment_id_map.setdefault(batch_number, []).append((experiment_id, row, col))
                        elif batch_number == 0:
                            print(f"Skipping experiment {experiment_id} as it's marked as '已经做完'.")
                        elif batch_number == -1:
                            print(f"Skipping experiment {experiment_id} as it's marked as '不用做'.")
                    else:
                        # 没有批次信息，默认为第1批，如 "1"
                        batch_number = 1
                        batch_name = f"第{batch_number}批"
                        print(f"Setting experiment {experiment_id} as {batch_name} (default).")
                        batch_to_experiment_id_map.setdefault(batch_number, []).append((experiment_id, row, col))
                else:
                    print(f"无法解析实验编号格式: {cell_str}")
    
    # 按批次顺序排序字典映射，也就是把字典映射按照键从小到大的顺序排序，排序后的字典格式为：
    '''
    {-1:[3,1,5],#表示编号为3,1和5的实验不用做
      0:[4,2,6],#表示编号为4,2和6的实验已做完
      1:[9],#表示编号为9的实验第1批做
      2:[10],#表示编号为10的实验第2批做
      ...
    }
    '''
    sorted_batch_to_experiment_id_map = {key: batch_to_experiment_id_map[key] for key in sorted(batch_to_experiment_id_map)}
    return sorted_batch_to_experiment_id_map
