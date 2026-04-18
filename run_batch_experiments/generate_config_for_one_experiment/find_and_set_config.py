
import openpyxl
from openpyxl.utils.cell import get_column_letter

# 定义一个辅助函数，用于在某个实验单元格的指定方向上查找实验参数，并设置该实验参数到实验配置Config中
#对于一个存储了实验编号的单元格，其上方和左方一定会有某个单元格存储着这个实验想要更改的实验参数
#我们搜集所有该实验单元格想要更改的实验参数
#然后把这些参数的具体数值更新到对应实验的配置Config中
#Config中未被更改的实验参数的数值使用Config初始化时的默认值
#由此，我们就确保了，不同的实验中，如实验A和实验B，仅有被修改的实验参数的数值不同
#而未被修改的实验参数的数值相同，均使用了Config的默认值
#由此达到了控制实验变量的目的，确保了实验结果的可靠
def find_and_set_config(config,experiment_setting_sheet,cell_address_to_hyperparam_map,direction, start_row, start_col):
        row, col = start_row, start_col  # 初始化某个实验编号的位置
        while True:
            # 根据方向查找行或列
            if direction == 'up':  # 向上移动
                row -= 1
            elif direction == 'left':  # 向左移动
                col -= 1
            
            # 边界检查，防止越界
            if row < 1 or col < 1:
                break  # 如果超出边界，则停止搜索
            
            # 构造当前单元格的地址（例如 "A1"）
            cell_address = f'{get_column_letter(col)}{row}'
            
            # 检查当前单元格地址是否存在于 cell_address_to_hyperparam_map中
            if cell_address in cell_address_to_hyperparam_map :
                config_key = cell_address_to_hyperparam_map [cell_address]  # 获取配置项名称
                cell_value = experiment_setting_sheet[cell_address].value  # 获取单元格的值
                
                # 如果单元格的值不为空，代表这个单元格存储着实验需要修改的变量值，我们修改其值到配置中
                if cell_value is not None:
                    # 处理字符串值：去除前后空格
                    if isinstance(cell_value, str):
                        cell_value = cell_value.strip()
                        # 尝试将字符串转换为适当的类型
                        try:
                            # 如果是布尔值（先检查，因为'true'中包含'e'）
                            if cell_value.lower() in ['true', 'false']:
                                cell_value = cell_value.lower() == 'true'
                            # 如果是整数
                            elif cell_value.isdigit() or (cell_value.startswith('-') and cell_value[1:].isdigit()):
                                cell_value = int(cell_value)
                            # 如果是浮点数
                            elif '.' in cell_value:
                                # 检查是否真的是浮点数格式（不是包含点的其他字符串）
                                parts = cell_value.split('.')
                                if len(parts) == 2 and parts[0].lstrip('-').isdigit() and parts[1].isdigit():
                                    cell_value = float(cell_value)
                            # 如果是科学计数法
                            elif 'e' in cell_value.lower():
                                # 检查科学计数法格式
                                parts = cell_value.lower().split('e')
                                if len(parts) == 2 and (parts[0].replace('.', '').replace('-', '').isdigit() or 
                                                       (parts[0].startswith('-') and parts[0][1:].replace('.', '').isdigit())) and \
                                   parts[1].lstrip('-').isdigit():
                                    cell_value = float(cell_value)
                        except (ValueError, AttributeError):
                            # 转换失败，保持原字符串
                            pass
                    
                    config.set(config_key, cell_value)  # 更新配置树
                    print(f"Set {config_key} to {cell_value}")  # 打印日志

        return config
