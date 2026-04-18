"""
配置管理器 - 主配置文件

此文件引用多个配置模块，构建完整的配置树。
配置数据被拆分为多个文件，便于管理和维护。
"""

# 导入各个配置模块
from .config_data.data import DATA_CONFIG
from .config_data.schedule import SCHEDULE_CONFIG
from .config_data.model import MODEL_CONFIG
from .config_data.federated import FEDERATED_CONFIG
from .config_data.attack import ATTACK_CONFIG


class ConfigTree:
    def __init__(self):
        """
        初始化配置管理器，并定义默认的 config_tree。
        配置树由多个模块化的配置部分组成。
        """
        self.config_tree = {
            "content": "Config",
            "type": "key",
            "children": [
                DATA_CONFIG,
                {
                    "content": "device",
                    "type": "key",
                    "children": [
                        {"content": "cuda:1", "type": "value"},
                    ]
                },
                SCHEDULE_CONFIG,
                MODEL_CONFIG,
                ATTACK_CONFIG,
                FEDERATED_CONFIG,
            ]
        }

    def update_choices_from_stack(self, stack):
        """
        根据栈中的完整路径，更新配置树中符合条件的键节点的 choice 属性。

        :param stack: 从根节点到目标节点的完整路径栈（仅包含 content）。
        """
        current_node = self.config_tree

        # 遍历栈中的路径内容，找到对应的节点并更新 choice 属性
        for i in range(1, len(stack) - 2):  # 从第二个元素开始，直到倒数第二个元素
            next_content = stack[i + 1]
            found = False
            for child in current_node.get("children", []):
                if child["content"] == stack[i] and child["type"] == "key":
                    found = True
                    # 更新 choice 属性为下一个节点的内容
                    if "choice" in child:
                        child["choice"] = next_content
                    break

            if not found:
                raise ValueError(f"Path segment not found or invalid type/choice: {stack[i]}")

            # 移动到下一个节点
            for child in current_node.get("children", []):
                if child["content"] == stack[i]:
                    current_node = child
                    break

    def check_unique_keys(self):
        """
        检查配置树中的所有 key 类型节点的 content 是否唯一。
        """
        seen_keys = set()

        def _check(node):
            if node["type"] == "key":
                key_content = node["content"]
                if key_content in seen_keys:
                    raise ValueError(f"Duplicate key found: {key_content}")
                seen_keys.add(key_content)

            for child in node.get("children", []):
                _check(child)

        _check(self.config_tree)

    def get(self, path, default=None):
        """
        获取配置树中指定路径的值。

        :param path: 路径字符串，格式为 "Config-Data_Config-Dataset"。
        :param default: 如果路径不存在，返回的默认值。
        :return: 节点的 content 值，如果路径不存在且提供了默认值，则返回默认值。
        """
        keys = path.split('-')

        def find_node(node, key):
            """递归查找节点"""
            if node["content"] == key:
                return node
            for child in node.get("children", []):
                result = find_node(child, key)
                if result:
                    return result
            return None

        # 查找路径的第一个节点
        first_key = keys[0]
        current_node = find_node(self.config_tree, first_key)

        if current_node is None:
            if default is not None:
                return default
            raise ValueError(f"First segment not found: {first_key} in path: {path}")

        # 遍历剩余的路径片段
        for i, key in enumerate(keys[1:], start=1):
            found = False
            for child in current_node.get("children", []):
                if child["content"] == key:
                    current_node = child
                    found = True
                    break
            if not found:
                if default is not None:
                    return default
                raise ValueError(f"Path not found at segment {i}: {key} in path: {path}")

        # 检查当前节点是否是键类型
        if current_node["type"] != "key":
            if default is not None:
                return default
            raise ValueError(f"Node at path {path} is not a key type")

        # 检查当前节点是否有 choice
        if "choice" in current_node:
            choices = current_node["choice"]
            if isinstance(choices, str):
                return choices
            selected_values = []
            for choice in choices:
                found = False
                for child in current_node["children"]:
                    if child["content"] == choice:
                        selected_values.append(child["content"])
                        found = True
                        break
                if not found:
                    if default is not None:
                        return default
                    raise ValueError(f"Choice not found in children: {choice}")
            return selected_values

        # 检查当前节点的子节点是否都是值类型
        value_children = [child for child in current_node["children"] if child["type"] == "value"]

        # 检查子节点是否唯一
        if len(value_children) != 1:
            if default is not None:
                return default
            raise ValueError(f"Expected exactly one value child, but found {len(value_children)} at path: {path}")

        # 返回唯一的值子节点的 content
        return value_children[0]["content"]

    def set(self, path, new_content):
        """
        设置配置树中指定路径的值。

        :param path: 路径字符串，格式为 "Config-Data_Config-Dataset"。
        :param new_content: 新的 content 值。
        """
        keys = path.split('-')
        stack = []  # 使用栈记录访问路径

        def find_node(node, key):
            """递归查找节点"""
            stack.append(node["content"])
            if node["content"] == key:
                return node
            for child in node.get("children", []):
                result = find_node(child, key)
                if result:
                    return result
            stack.pop()
            return None

        # 查找路径的第一个节点
        first_key = keys[0]
        current_node = find_node(self.config_tree, first_key)

        if current_node is None:
            raise ValueError(f"First segment not found: {first_key} in path: {path}")

        # 遍历剩余的路径片段
        for i, key in enumerate(keys[1:], start=1):
            found = False
            for child in current_node.get("children", []):
                if child["content"] == key:
                    stack.append(child["content"])
                    current_node = child
                    found = True
                    break
            if not found:
                raise ValueError(f"Path not found at segment {i}: {key} in path: {path}")

        self.update_choices_from_stack(stack)

        # 检查当前节点是否是键类型
        if current_node["type"] != "key":
            raise ValueError(f"Node at path {path} is not a key type")

        # 检查当前节点是否有 choice
        if "choice" in current_node:
            if isinstance(new_content, list):
                valid_choices = [child["content"] for child in current_node["children"] if child["type"] == "value"]
                for choice in new_content:
                    if choice not in valid_choices:
                        raise ValueError(f"Invalid choice: {choice}. Valid choices are: {valid_choices}")

                # 设置新的 choice
                current_node["choice"] = new_content
                return
            elif isinstance(new_content, str):
                valid_choices = [child["content"] for child in current_node["children"] if child["type"] == "value"]
                if new_content not in valid_choices:
                    raise ValueError(f"Invalid choice: {new_content}. Valid choices are: {valid_choices}")

                # 设置新的 choice
                current_node["choice"] = new_content
                return
            else:
                raise ValueError(f"Invalid type for new_content: {type(new_content)}. Expected list or str.")

        # 检查当前节点的子节点是否都是值类型
        value_children = [child for child in current_node["children"] if child["type"] == "value"]

        # 检查子节点是否唯一
        if len(value_children) != 1:
            raise ValueError(f"Expected exactly one value child, but found {len(value_children)} at path: {path}")

        # 设置唯一的值子节点的 content
        value_children[0]["content"] = new_content


# 创建全局配置实例
config = ConfigTree()

if __name__ == "__main__":
    # 测试用例 3: 更新 fedprox_mu 的值
    config = ConfigTree()
    config.set("fedprox_mu", 0.02)
    print("After updating fedprox_mu to 0.02:")
    print(config.config_tree)
