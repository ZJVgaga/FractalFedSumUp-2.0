
from .reverse_hyperparam_to_cell_address_map import reverse_hyperparam_to_cell_address_map
from .find_and_set_config import find_and_set_config
import openpyxl
from openpyxl.utils.cell import get_column_letter


# 定义一个函数，对于某一个单个的实验，从实验设置表中提取实验参数值并更新该实验的配置
def generate_config_for_one_experiment(experiment_setting_sheet, hyperparam_to_cell_address_map, experiment_id, current_row, current_col,config):
    
    """
    从工作表中提取配置值，并根据单元格地址映射更新配置树。

    参数:
        experiment_setting_sheet (openpyxl.Worksheet): 输入的工作表对象。
        hyperparam_to_cell_address_map (dict): 单元格地址到配置项名称的映射字典。
                         格式：{"A1": "config_key_1", "B2": "config_key_2", ...}
        experiment_id (int): 当前实验的索引或 ID。
        current_row (int): 当前单元格所在的行号。
        current_col (int): 当前单元格所在的列号。

    返回:
        config(ConfigTree):指定实验的配置树 
    """
    # 翻转映射
    cell_address_to_hyperparam_map = reverse_hyperparam_to_cell_address_map(hyperparam_to_cell_address_map)
    """
    cell_address_to_hyperparam_map格式：
            {"A1": "config_key_1", "A2": "config_key_1", ...}
    """
    
    # 向上回溯，查找上方单元格中的配置项
    config=find_and_set_config(config,experiment_setting_sheet,cell_address_to_hyperparam_map,'up', current_row, current_col)
    
    # 向左回溯，查找左侧单元格中的配置项
    config=find_and_set_config(config,experiment_setting_sheet,cell_address_to_hyperparam_map,'left', current_row, current_col)
    
    # 设置当前实验的索引或 ID 到配置树中
    config.set("experiment_index", experiment_id)
    return config


