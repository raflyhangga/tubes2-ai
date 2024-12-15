import pandas as pd
import numpy as np
import json

class TreeNode:
    def __init__(self, node: str, branch: dict, default: str = None):
        self.node = node        # attribute name
        self.branch = branch    # dictionary of branch (key: branch value, value: TreeNode)
        self.default = default  # default value

    def __str__(self):
        return f"TreeNode({self.node}, {self.branch}, {self.default})"
    
    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        return self.__dict__[name]

    def print_tree(self, level: int = 0) -> None:
        """function for printing the tree"""
        for key, value in self.branch.items():
            print("    " * level, end="")
            print(f'{self.node} {key}')
            if value is not None:
                value.print_tree(level + 1)
    
    def tree_to_dict(self) -> dict:
        """function for converting the tree into nested dictionary
        
        Returns:
            dict: nested dictionary
        """
        data = {
            "node": self.node,
            "branch": {
                key: (value.tree_to_dict() if value is not None else None) for key, value in self.branch.items()
            },
            "default": self.default
        }
        return data
    
    def dict_to_tree(self, dictionary: dict) -> "TreeNode":
        """function for converting nested dictionary into tree
        
        Args:
            dictionary (dict): nested dictionary
        
        Returns:
            TreeNode: tree
        """
        self.node = dictionary["node"]
        self.branch = {
            key: (TreeNode("", {}).dict_to_tree(value) if value is not None else None) for key, value in dictionary["branch"].items()
        }
        self.default = dictionary["default"]
        return self

class IterativeDichotomiser3:
    def __init__(self, max_depth: int = 15):
        self.max_depth = max_depth
        self.tree : TreeNode = None
        self.list_attribute_names : list = None
        self.target_attribute : str = None
        self.list_target_attribute_values : list = None

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        return self.__dict__[name]
    

    """ 
        Function for training ID3 model 
    """
    def fit(self, data: dict, target_name: str) -> None:
        """function for fitting the model

        Args:
            data (dict): data
            target_name (str): target attribute
        """
        self.list_attribute_names = data.columns.tolist()
        try:
            self.list_attribute_names.remove(target_name)
            self.list_attribute_names.remove("id")
        except:
            pass
        self.target_attribute = target_name
        self.list_target_attribute_values = data[target_name].unique().tolist()

        self.tree = self.train(data, self.list_attribute_names, {}, self.max_depth)
    
    def train(self, data: dict, attribute: list[str], parent_data: dict, depth: int) -> TreeNode:
        """function for training ID3 model

        Args:
            data (dict): data for current node
            attribute (list[str]): list of remaining attribute
            parent_data (dict): data from parent node
            depth (int): depth of the tree

        pre-condition:
            all attributes in the data are numeric (should have been done in preprocessing)
            
        Returns:
            TreeNode: tree node
        """
        if depth == 0:
            return TreeNode(self.target_attribute, 
                            {"= " + str(self.plurality_value(parent_data, self.target_attribute)): None},
                            self.plurality_value(parent_data, self.target_attribute))
        elif len(data) == 0:
            return TreeNode(self.target_attribute, 
                            {"= " + str(self.plurality_value(parent_data, self.target_attribute)): None}, 
                            self.plurality_value(parent_data, self.target_attribute))
        elif self.is_homogeneous(data):
            return TreeNode(self.target_attribute, 
                            {"= " + (data[self.target_attribute].iloc[0]): None}, 
                            data[self.target_attribute].iloc[0])
        elif len(attribute) == 0:
            return TreeNode(self.target_attribute, 
                            {"= " + (self.plurality_value(data, self.target_attribute)): None}, 
                            self.plurality_value(data, self.target_attribute))
        else:
            # find the best attribute
            best_attribute, break_point = self.find_best_attribute(data, attribute)
            tree = TreeNode(best_attribute, 
                            {}, 
                            self.plurality_value(data, self.target_attribute))
            # print(f'{depth} Best Attribute: {best_attribute}, Break Point: {break_point}')

            # split the data
            new_data_less_than = data[data[best_attribute] < break_point]
            new_data_greater_than = data[data[best_attribute] >= break_point]
            
            # remove the best attribute from the list
            new_attribute = attribute.copy()                    # remaining attribute
            new_attribute.remove(best_attribute)

            # train the tree recursively
            tree.branch["< " + str(break_point)] = self.train(new_data_less_than, new_attribute, data, depth - 1)
            tree.branch[">= " + str(break_point)] = self.train(new_data_greater_than, new_attribute, data, depth - 1)
            
            return tree

    def plurality_value(self, data: dict, attribute: str) -> str|float:
        """function for finding the most common value in data

        Args:
            data (dict): data
            attribute (str): attribute

        Returns:
            str|float: most common value
        """
        return data[attribute].value_counts().idxmax()

    def is_homogeneous(self, data: dict) -> bool:
        """function for checking if data is homogeneous

        Args:
            data (dict): data

        Returns:
            bool: True if data is homogeneous, otherwise False
        """
        return len(data[self.target_attribute].unique()) == 1

    def entropy(self, data: dict) -> float:
        """function for calculating entropy
                entropy = -sum(p * log2(p))
                p = frequency of each class / total data

        Args:
            data (dict): data

        Returns:
            float: entropy
        """
        count_unique = data[self.target_attribute].value_counts()
        length_data = len(data)
        entropy = -np.sum(count_unique / length_data * np.log2(count_unique / length_data))
        return (entropy if (entropy != 0) else 0)
        
    def information_gain(self, data: dict, attribute: str, break_point: float = None) -> float:
        """function for calculating information gain
                information gain = entropy(parent) - sum(entropy(children) * (len(child) / len(parent))

        Args:
            data (dict): data
            attribute (str): attribute to be calculated
            break_point (float, optional): break point for continuous data. Defaults to None.

        Returns:
            float: information gain
        """
        entropy_parent = self.entropy(data)
        entropy_children = 0
        data_less_than = data[data[attribute] < break_point]
        data_greater_than = data[data[attribute] >= break_point]
        entropy_children = (len(data_less_than) / len(data) * self.entropy(data_less_than)) + (len(data_greater_than) / len(data) * self.entropy(data_greater_than))
        return entropy_parent - entropy_children

    def find_best_attribute(self, data: dict, attribute: list[str]) -> tuple:
        """function for finding the best attribute

        Args:
            data (dict): data
            attribute (list[str]): list of attribute

        Returns:
            tuple: best attribute and break point (if the best attribute is continuous)
        """
        max_gain = -1
        best_attribute = None
        break_point = None
        for attr in attribute:
            temp_break_point = self.find_break_point(data, attr)
            gain = self.information_gain(data, attr, temp_break_point)
            if gain > max_gain:
                max_gain = gain
                best_attribute = attr
                break_point = temp_break_point
        return best_attribute, break_point

    def find_break_point(self, data: dict, attribute: str) -> float:
        """function for finding break point of continuous data

        Args:
            data (dict): data
            attribute (str): attribute

        Returns:
            float: break point
        """
        best_break_point = (data[attribute].iloc[0] + data[attribute].iloc[1]) / 2
        max_gain = self.information_gain(data, attribute, best_break_point)
        data = data.sort_values(by=attribute)
        # print(f'data: {data}')
        # print(f'atribut: {attribute}')
        for i in range(0, len(data) - 2, 5000):
            # find two consecutive data with different target attribute (candiate for break point)
            if data[self.target_attribute].iloc[i] != data[self.target_attribute].iloc[i + 1]:
                break_point = (data[attribute].iloc[i] + data[attribute].iloc[i + 1]) / 2
                gain = self.information_gain(data, attribute, break_point)
                # print(f'Break Point: {break_point}, Gain: {gain}')
                if gain > max_gain:
                    max_gain = gain
                    best_break_point = break_point
        # print(f'Best Break Point: {best_break_point}, Gain: {max_gain}')
        return best_break_point



    """
        Function for predicting data
    """
    def predict_row(self, unseen_data: dict) -> str:
        """function for predicting unseen data

        Args:
            unseen_data (dict): unseen data

        Returns:
            str: prediction
        """
        prediction = self.predict_recursively(unseen_data, self.tree)
        return prediction
        
    def predict_recursively(self, unseen_data: dict, tree: TreeNode) -> str:
        """function for predicting unseen data recursively

        Args:
            unseen_data (dict): unseen data
            tree (TreeNode): tree

        Returns:
            str: prediction
        """
        prediction = None
        current_atribute = tree.node
        if current_atribute == self.target_attribute:
            prediction = tree.branch.keys().__iter__().__next__().split(" ")[1]
        else:
            for key, value in tree.branch.items():
                if key.split(" ")[0] == "=":
                    if unseen_data[current_atribute] == key.split(" ")[1]:
                        if value is None:
                            prediction = key.split(" ")[1]
                        else:
                            prediction = self.predict_recursively(unseen_data, value)
                elif key.split(" ")[0] == "<":
                    if unseen_data[current_atribute] < float(key.split(" ")[1]):
                        if value is None:
                            prediction = key.split(" ")[1]
                        else:
                            prediction = self.predict_recursively(unseen_data, value)
                elif key.split(" ")[0] == ">=":
                    if unseen_data[current_atribute] >= float(key.split(" ")[1]):
                        if value is None:
                            prediction = key.split(" ")[1]
                        else:
                            prediction = self.predict_recursively(unseen_data, value)
        return (prediction if prediction is not None else tree.default)

    def predict(self, unseen_data: pd.DataFrame) -> np.ndarray:
        """function for predicting unseen data in batch

        Args:
            unseen_data (pd.DataFrame): unseen data

        Returns:
            list: list of prediction
        """
        predictions = []
        for i in range(len(unseen_data)):
            predictions.append(self.predict_row(unseen_data.iloc[i]))
        return np.array(predictions)


    """
        Function for printing the tree
    """
    def print_tree(self) -> None:
        """function for printing the tree"""
        self.tree.print_tree()


    """
        Function for saving and loading the tree
    """
    def save_tree(self, filename: str) -> None:
        """function for saving the tree into file .json
        
        Args:
            filename (str): filename
        """
        try:
            with open(filename, "w") as file:
                json.dump(self.tree.tree_to_dict(), file)
            print(f"Tree saved to {filename}")
        except Exception as e:
            print(f"Failed to save tree. Error: {e}")
    
    def load_tree(self, filename: str) -> None:
        """function for loading the tree from file .json

        Args:
            filename (str): filename
        """
        try:
            with open(filename, "r") as file:
                dict_tree = json.load(file)
            self.tree = TreeNode("", {}).dict_to_tree(dict_tree)
            print(f"Tree loaded from {filename}")
        except Exception as e:
            print(f"Failed to load tree. Error: {e}")