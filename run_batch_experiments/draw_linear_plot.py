import openpyxl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, rgb_to_hsv, hsv_to_rgb
import os
import re

def parse_value_with_error(value):
    """解析“平均值±标准差”格式的字符串或直接处理数值"""
    # 如果值是None，返回None
    if value is None:
        return None
    
    # 如果值是字符串类型
    if isinstance(value, str):
        match = re.match(r"([-\d.]+)±([-\d.]+)", value)
        if match:
            mean = float(match.group(1))
            std = float(match.group(2))
            return mean, std
        else:
            # 尝试直接转换为浮点数
            try:
                mean = float(value)
                return mean, 0.0
            except ValueError:
                raise ValueError(f"无法解析值: {value}")
    # 如果值是数值类型（int, float等）
    elif isinstance(value, (int, float)):
        return float(value), 0.0
    else:
        raise ValueError(f"不支持的数据类型: {type(value)}")

def desaturate_color(color, amount=0.5, alpha=1.0):
    """降低颜色饱和度并添加灰度"""
    # 将RGB转换为HSV以调整饱和度
    hsv = rgb_to_hsv(color[:3])
    # 减少饱和度
    hsv_desaturated = (hsv[0], hsv[1] * amount, hsv[2])
    # 转换回RGB
    rgb_desaturated = hsv_to_rgb(hsv_desaturated)
    # 返回包含指定透明度的新RGBA颜色
    return tuple(min(max(c, 0), 1) for c in rgb_desaturated) + (alpha,)

def plot_sheets_with_pattern(workbook_path, pattern, font_size=12, alpha=0.8):
    # 加载工作簿
    workbook = openpyxl.load_workbook(workbook_path)
    
    # 创建目录以保存图像（如果不存在）
    save_dir = os.path.splitext(workbook_path)[0] + "_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"开始处理文件: {workbook_path}")
    
    # 遍历所有工作表
    for sheet_name in workbook.sheetnames:
        if pattern in sheet_name:
            sheet = workbook[sheet_name]
            
            print(f"正在处理工作表: {sheet_name}")
            
            # 获取x轴数据和标签
            x_label = sheet['A1'].value
            x_values_all = [cell.value for cell in sheet['A'][1:] if cell.value is not None]  # A列从A2开始
            
            # 获取y轴标签
            y_label = sheet['B1'].value
            
            # 获取折线名称和颜色
            line_names = []
            colors = []
            for col in range(3, sheet.max_column + 1):  # C列开始到最后
                cell = sheet.cell(row=1, column=col)
                if cell.value is not None:  # 只有非空的单元格作为折线名称
                    line_names.append(cell.value)
                    color_hex = cell.fill.start_color.index
                    # 处理颜色可能是ARGB格式的问题
                    if len(color_hex) > 6 and color_hex.startswith('FF'):  # 去除alpha部分
                        color_hex = color_hex[2:]
                    rgba_color = tuple(int(color_hex[i:i+2], 16)/255 for i in (0, 2, 4)) + (1,)  # 添加透明度
                    colors.append(desaturate_color(rgba_color, alpha=alpha))
            
            print(f"找到 {len(line_names)} 条折线.")
            
            # 绘制折线图
            plt.figure(figsize=(10, 6))
            for idx, name in enumerate(line_names):
                y_data = [parse_value_with_error(cell.value) for cell in sheet[f'{chr(64 + idx + 3)}'][1:] if cell.value is not None]
                means = [data[0] for data in y_data]
                errors = [data[1] for data in y_data]
                
                # 确保 x_values 和 y_values 长度一致
                x_values = x_values_all[:len(means)]
                
                plt.errorbar(x_values, means, yerr=errors, label=name, color=colors[idx], fmt='-o', alpha=alpha)
            
            plt.xlabel(x_label, fontsize=font_size)
            plt.ylabel(y_label, fontsize=font_size)
            plt.title("", fontsize=font_size)
            plt.legend(fontsize=font_size, framealpha=0.5)  # 图例半透明
            
            # 设置全局字体大小
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            
            # 保存图表
            image_path = os.path.join(save_dir, f"{sheet_name}.png")
            plt.savefig(image_path)
            plt.close()  # 关闭当前图表，以便绘制下一个图表时不会叠加在前一个图表上
            print(f"已保存图表至: {image_path}")

# 使用函数
plot_sheets_with_pattern('guess-slide-mechenism-last_multiple-超精细.xlsx', '结果展示图_', font_size=18, alpha=0.8)
