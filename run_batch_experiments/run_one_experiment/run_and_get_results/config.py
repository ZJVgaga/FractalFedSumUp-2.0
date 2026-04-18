"""
Configuration Manager - Main Configuration File

This file references multiple configuration modules to build a complete configuration tree.
Configuration data is split into multiple files for easier management and maintenance.
"""

# Import various configuration modules
from .config_data.data import DATA_CONFIG
from .config_data.schedule import SCHEDULE_CONFIG
from .config_data.model import MODEL_CONFIG
from .config_data.federated import FEDERATED_CONFIG
from .config_data.attack import ATTACK_CONFIG


class ConfigTree:
    def __init__(self):
        """
        Initialize the configuration manager and define the default config_tree.
        The configuration tree consists of multiple modular configuration parts.
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
        Update the choice attribute for eligible key nodes in the configuration tree based on the full path in the stack.

        :param stack: The complete path stack from the root node to the target node (contains only content).
        """
        current_node = self.config_tree

        # Traverse the path content in the stack, find the corresponding node, and update the choice attribute
        for i in range(1, len(stack) - 2):  # Start from the second element until the second-to-last element
            next_content = stack[i + 1]
            found = False
            for child in current_node.get("children", []):
                if child["content"] == stack[i] and child["type"] == "key":
                    found = True
                    # Update the choice attribute to the content of the next node
                    if "choice" in child:
                        child["choice"] = next_content
                    break

            if not found:
                raise ValueError(f"Path segment not found or invalid type/choice: {stack[i]}")

            # Move to the next node
            for child in current_node.get("children", []):
                if child["content"] == stack[i]:
                    current_node = child
                    break

    def check_unique_keys(self):
        """
        Check if the content of all key-type nodes in the configuration tree is unique.
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
        Get the value at the specified path in the configuration tree.

        :param path: Path string in the format "Config-Data_Config-Dataset".
        :param default: Default value to return if the path does not exist.
        :return: The content value of the node, or the default value if the path does not exist and a default is provided.
        """
        keys = path.split('-')

        def find_node(node, key):
            """Recursively find a node"""
            if node["content"] == key:
                return node
            for child in node.get("children", []):
                result = find_node(child, key)
                if result:
                    return result
            return None

        # Find the first node of the path
        first_key = keys[0]
        current_node = find_node(self.config_tree, first_key)

        if current_node is None:
            if default is not None:
                return default
            raise ValueError(f"First segment not found: {first_key} in path: {path}")

        # Traverse the remaining path segments
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

        # Check if the current node is a key type
        if current_node["type"] != "key":
            if default is not None:
                return default
            raise ValueError(f"Node at path {path} is not a key type")

        # Check if the current node has a choice
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

        # Check if all children of the current node are value types
        value_children = [child for child in current_node["children"] if child["type"] == "value"]

        # Check if the child node is unique
        if len(value_children) != 1:
            if default is not None:
                return default
            raise ValueError(f"Expected exactly one value child, but found {len(value_children)} at path: {path}")

        # Return the content of the unique value child node
        return value_children[0]["content"]

    def set(self, path, new_content):
        """
        Set the value at the specified path in the configuration tree.

        :param path: Path string in the format "Config-Data_Config-Dataset".
        :param new_content: New content value.
        """
        keys = path.split('-')
        stack = []  # Use a stack to record the access path

        def find_node(node, key):
            """Recursively find a node"""
            stack.append(node["content"])
            if node["content"] == key:
                return node
            for child in node.get("children", []):
                result = find_node(child, key)
                if result:
                    return result
            stack.pop()
            return None

        # Find the first node of the path
        first_key = keys[0]
        current_node = find_node(self.config_tree, first_key)

        if current_node is None:
            raise ValueError(f"First segment not found: {first_key} in path: {path}")

        # Traverse the remaining path segments
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

        # Check if the current node is a key type
        if current_node["type"] != "key":
            raise ValueError(f"Node at path {path} is not a key type")

        # Check if the current node has a choice
        if "choice" in current_node:
            if isinstance(new_content, list):
                valid_choices = [child["content"] for child in current_node["children"] if child["type"] == "value"]
                for choice in new_content:
                    if choice not in valid_choices:
                        raise ValueError(f"Invalid choice: {choice}. Valid choices are: {valid_choices}")

                # Set the new choice
                current_node["choice"] = new_content
                return
            elif isinstance(new_content, str):
                valid_choices = [child["content"] for child in current_node["children"] if child["type"] == "value"]
                if new_content not in valid_choices:
                    raise ValueError(f"Invalid choice: {new_content}. Valid choices are: {valid_choices}")

                # Set the new choice
                current_node["choice"] = new_content
                return
            else:
                raise ValueError(f"Invalid type for new_content: {type(new_content)}. Expected list or str.")

        # Check if all children of the current node are value types
        value_children = [child for child in current_node["children"] if child["type"] == "value"]

        # Check if the child node is unique
        if len(value_children) != 1:
            raise ValueError(f"Expected exactly one value child, but found {len(value_children)} at path: {path}")

        # Set the content of the unique value child node
        value_children[0]["content"] = new_content


# Create a global configuration instance
config = ConfigTree()

if __name__ == "__main__":
    # Test case 3: Update the value of fedprox_mu
    config = ConfigTree()
    config.set("fedprox_mu", 0.02)
    print("After updating fedprox_mu to 0.02:")
    print(config.config_tree)