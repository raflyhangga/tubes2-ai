import pandas as pd
import numpy as np

class TreeNode:
    def __init__(self, node: str, branch: dict):
        self.node = node        # attribute name
        self.branch = branch    # dictionary of branch (key: branch value, value: TreeNode)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        return self.__dict__[name]

    def print_tree(self, level: int = 0):
        """function for printing the tree"""
        for key, value in self.branch.items():
            print("    " * level, end="")
            print(f'{self.node} {key}')
            if value is not None:
                value.print_tree(level + 1)

class ID3:
    def __init__(self):
        self.tree : TreeNode = None
        self.list_attribute_names : list = None
        self.list_attribute_types : dict = None
        self.target_attribute : str = None
        self.list_target_attribute_values : list = None

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        return self.__dict__[name]
    


    """ 
        Function for training ID3 model 
    """
    def train(self, data: dict, attribute: list[str], parent_data: dict) -> TreeNode:
        """function for training ID3 model

        Args:
            data (dict): data for current node
            attribute (list[str]): list of remaining attribute
            parent_data (dict): data from parent node
        """
        if len(data) == 0:
            return TreeNode(self.target_attribute, {"= " + str(self.plurality_value(parent_data, self.target_attribute)): None})
        elif self.is_homogeneous(data):
            return TreeNode(self.target_attribute, {"= " + (data[self.target_attribute].unique()[0]): None})
        elif len(attribute) == 0:
            return TreeNode(self.target_attribute, {"= " + (self.plurality_value(data, self.target_attribute)): None})
        else:
            best_attribute, break_point = self.find_best_attribute(data, attribute)
            tree = TreeNode(best_attribute, {})
            if self.list_attribute_types[best_attribute] == "categorical":
                for value in data[best_attribute].unique():
                    new_data = data[data[best_attribute] == value]  # data with best_attribute = value
                    new_attribute = attribute.copy()                # remaining attribute
                    new_attribute.remove(best_attribute)
                    tree.branch["= " + value] = self.train(new_data, new_attribute, data)
            else:
                new_data_less_than = data[data[best_attribute] < break_point]
                new_data_greater_than = data[data[best_attribute] >= break_point]
                new_attribute = attribute.copy()                    # remaining attribute
                new_attribute.remove(best_attribute)
                tree.branch["< " + str(break_point)] = self.train(new_data_less_than, new_attribute, data)
                tree.branch[">= " + str(break_point)] = self.train(new_data_greater_than, new_attribute, data)
            return tree

    def plurality_value(self, data: dict, attribute: str) -> str|int|float|bool|None:
        """function for finding the most common value in data

        Args:
            data (dict): data
            attribute (str): attribute

        Returns:
            str|int|float|bool|None: the most common value in data
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
        entropy = -np.sum(data[self.target_attribute].value_counts() / len(data) * np.log2(data[self.target_attribute].value_counts() / len(data)))
        return (entropy if (entropy != 0) else 0)
        
    def information_gain(self, data: dict, attribute: str, break_point: float = None) -> float:
        """function for calculating information gain
                information gain = entropy(parent) - sum(entropy(children) * (len(child) / len(parent))

        Args:
            data (dict): data
            attribute (str): attribute to be calculated

        Returns:
            float: information gain
        """
        entropy_parent = self.entropy(data)
        entropy_children = 0
        if self.list_attribute_types[attribute] == "categorical":
            for value in data[attribute].unique():
                entropy_children += len(data[data[attribute] == value]) / len(data) * self.entropy(data[data[attribute] == value])
        else:
            data_less_than = data[data[attribute] < break_point]
            data_greater_than = data[data[attribute] >= break_point]
            entropy_children = len(data_less_than) / len(data) * self.entropy(data_less_than) + len(data_greater_than) / len(data) * self.entropy(data_greater_than)
        return entropy_parent - entropy_children

    def find_best_attribute(self, data: dict, attribute: list[str]) -> tuple:
        """function for finding the best attribute

        Args:
            data (dict): data
            attribute (list[str]): list of attribute

        Returns:
            tuple: best attribute and break point (if the best attribute is continuous)
        """
        max_gain = 0
        best_attribute = None
        break_point = None
        for attr in attribute:
            if self.list_attribute_types[attr] == "categorical":
                gain = self.information_gain(data, attr)
                if gain > max_gain:
                    max_gain = gain
                    best_attribute = attr
            else:
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
        max_gain = 0
        best_break_point = None
        data = data.sort_values(by=attribute)
        # print(f'data: {data}')
        # print(f'atribut: {attribute}')
        for i in range(len(data) - 2):
            # find two consecutive data with different target attribute (candiate for break point)
            if data[self.target_attribute].iloc[i] != data[self.target_attribute].iloc[i + 1]:
                break_point = (data[attribute].iloc[i] + data[attribute].iloc[i + 1]) / 2
                gain = self.information_gain(data, attribute, break_point)
                # print(f'Break Point: {break_point}, Gain: {gain}')
                if gain > max_gain:
                    max_gain = gain
                    best_break_point = break_point
        return best_break_point



    """
        Function for predicting data
    """
    def predict(self, unseen_data: dict) -> str:
        """function for predicting unseen data

        Args:
            unseen_data (dict): unseen data

        Returns:
            str: prediction
        """
        prediction = self.predict_recursively(unseen_data, self.tree)
        return (prediction if prediction is not None else "Unknown") 
        
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
        return prediction



    """
        Function for printing the tree
    """
    def print_tree(self) -> None:
        """function for printing the tree"""
        self.tree.print_tree()



    """
        Function for saving and loading the tree
    """
    def load_tree(self, filename: str):
        """function for loading the tree

        Args:
            filename (str): filename
        """
        print("Ini load_tree")
        pass

    def save_tree(self, filename: str):
        """function for saving the tree
        
        Args:
            filename (str): filename
        """
        



if __name__ == "__main__":
    example_data1 = {
        "Day" : ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14"],
        "Outlook" : ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
        "Temperature1" : ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
        "Temperature2" : [10, 11, 60, 70, 68, 12, 64, 11, 69, 75, 75, 72, 81, 11],
        "Humidity" : ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
        "Wind" : ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
        "PlayTennis" : ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
    }

    example_data2 = {
        "Day" : ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14"],
        "Outlook" : ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
        "Temperature1" : ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
        "Temperature2" : [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
        "Humidity" : ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
        "Wind" : ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
        "PlayTennis" : ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"]
    }

    list_attribute_names1 : list = ["Outlook", "Temperature1", "Humidity", "Wind"]
    list_attribute_names2 : list = ["Outlook", "Temperature2", "Humidity", "Wind"]
    list_attribute_types : dict = {"Outlook": "categorical", "Temperature1": "categorical", "Temperature2": "numerical", "Humidity": "categorical", "Wind": "categorical"}
    target_attribute : str = "PlayTennis"
    list_target_attribute_values : list = ["Yes", "No"]

    df1 = pd.DataFrame(example_data1)
    df2 = pd.DataFrame(example_data2)
    # print(df1)
    # print(df2)

    id3 = ID3()
    id3.list_attribute_names = list_attribute_names1
    id3.list_attribute_types = list_attribute_types
    id3.target_attribute = target_attribute
    id3.list_target_attribute_values = list_target_attribute_values

    print("======================= ID3 Model =======================")
    print(id3.list_attribute_names)
    print(id3.list_attribute_types)
    print(id3.target_attribute)
    print(id3.list_target_attribute_values)
    print(id3.tree)
    print("")
    
    """
        Unit Test
    """
    # test plurality_value
    print("======================= Test plurality_value =======================")
    print(f'Plurality Value Outlook: {id3.plurality_value(df1, "Outlook")}')          # Sunny
    print(f'Plurality Value Temperature: {id3.plurality_value(df1, "Temperature1")}') # Mild
    print(f'Plurality Value Humidity: {id3.plurality_value(df1, "Humidity")}')        # High
    print(f'Plurality Value Wind: {id3.plurality_value(df1, "Wind")}')                # Weak
    print(f'Plurality Value PlayTennis: {id3.plurality_value(df1, "PlayTennis")}')    # Yes
    print("")


    # test is_homogeneous
    print("======================= Test is_homogeneous =======================")
    print(f'Is Homogeneous df1: {id3.is_homogeneous(df1)}')  # False
    print(f'Is Homogeneous df2: {id3.is_homogeneous(df2)}')  # True
    print("")


    # test entropy
    print("======================= Test entropy =======================")
    # entropy for all data
    print(f'Entropy df1: {id3.entropy(df1)}')  # 0.940286
    print(f'Entropy df2: {id3.entropy(df2)}')  # 0

    # entropy for Outlook = Sunny
    df_sunny = df1[df1["Outlook"] == "Sunny"]
    print(f'Entropy df_sunny: {id3.entropy(df_sunny)}')  # 0.970951

    # entropy for Outlook = Overcast
    df_overcast = df1[df1["Outlook"] == "Overcast"]
    print(f'Entropy df_overcast: {id3.entropy(df_overcast)}')  # 0

    # entropy for Outlook = Rain
    df_rain = df1[df1["Outlook"] == "Rain"]
    print(f'Entropy df_rain: {id3.entropy(df_rain)}')  # 0.970951
    print("")


    # test information_gain
    print("======================= Test information_gain =======================")
    print(f'Information Gain Outlook: {id3.information_gain(df1, "Outlook")}')          # 0.246749
    print(f'Information Gain Temperature: {id3.information_gain(df1, "Temperature1")}') # 0.029223
    print(f'Information Gain Humidity: {id3.information_gain(df1, "Humidity")}')        # 0.151835
    print(f'Information Gain Wind: {id3.information_gain(df1, "Wind")}')                # 0.048127
    print("")


    # test find_best_attribute
    print("======================= Test find_best_attribute =======================")
    print(f'Best Attribute: {id3.find_best_attribute(df1, ["Outlook", "Temperature1", "Humidity", "Wind"])}')  # Outlook
    print("")


    # test break_point
    print("======================= Test break_point =======================")
    data3 = {
        "Temperature2": [40, 48, 60, 72, 80, 90],
        "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No"]
    }
    df3 = pd.DataFrame(data3)
    print(f'Best Break Point: {id3.find_break_point(df3, "Temperature2")}')
    print("")
    # Best Break Point: 54.0, Gain: 0.459148
    # Break Point: 54.0, Gain: 0.459148
    # Break Point: 85.0, Gain: 0.190875


    # test train
    print("======================= Test train =======================")
    id3.tree = id3.train(df1, id3.list_attribute_names, {})
    id3.print_tree()
    print("")
    # id3.list_attribute_names = list_attribute_names2
    # id3.tree = id3.train(df1, id3.list_attribute_names, {})
    # id3.print_tree()
    # print("")

    # test predict
    print("======================= Test predict =======================")
    for i in range(len(df1)):
        print(f'Prediction {i}: {id3.predict(df1.iloc[i])}')
