import pandas as pd
import numpy as np

class Node:
    def __init__(self, attribute: str, value: str|int|float|bool, children: dict):
        self.attribute = attribute
        self.value = value
        self.children = children

class ID3:
    def __init__(self):
        self.tree : Node = None
        self.list_attribute_names : list = ["Outlook", "Temperature", "Humidity", "Wind"] # None       HARDCODED
        self.list_attribute_types : dict = {"Outlook": "categorical", "Temperature": "categorical", "Temperature2": "numerical", "Humidity": "categorical", "Wind": "categorical"} # None       HARDCODED
        self.target_attribute : str = "PlayTennis" # None       HARDCODED
        self.target_attribute_type : str = None
        self.list_target_attribute_values : list = ["Yes", "No"] # None       HARDCODED

    
    """ 
        Function for training ID3 model 
    """
    def train(self, data: dict, attribute: list[str], parent_data: dict) -> None:
        """function for training ID3 model

        Args:
            data (dict): data for current node
            attribute (list[str]): list of remaining attribute
            parent_data (dict): data from parent node
        """
        print("Ini train")
        pass

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

    def find_best_attribute(self, data: dict, attribute: list[str]) -> str:
        """function for finding the best attribute

        Args:
            data (dict): data
            attribute (list[str]): list of attribute

        Returns:
            str: the best attribute
        """
        return max(attribute, key=lambda x: self.information_gain(data, x))

    def break_point(self, data: dict, attribute: str) -> dict:
        """function for finding break point of continuous data

        Args:
            data (dict): data
            attribute (str): attribute

        Returns:
            dict: break point
        """
        max_gain = 0
        best_break_point = None
        print(f'atribut: {attribute}')
        for i in range(len(data) - 1):
            # find two consecutive data with different target attribute (candiate for break point)
            if data[self.target_attribute][i] != data[self.target_attribute][i + 1]:
                break_point = (data[attribute][i] + data[attribute][i + 1]) / 2
                gain = self.information_gain(data, attribute, break_point)
                print(f'Break Point: {break_point}, Gain: {gain}')
                if gain > max_gain:
                    max_gain = gain
                    best_break_point = break_point
        return best_break_point

    def set_attribute_type(self, data: dict) -> str:
        """function for getting attribute type

        Args:
            data (dict): data
            attribute (str): attribute

        Returns:
            str: attribute type
        """
        print("Ini set_attribute_type")
        pass

    def set_target_attribute_values(self, data: dict) -> list:
        """function for getting target attribute values

        Args:
            data (dict): data

        Returns:
            list: target attribute values
        """
        print("Ini set_target_attribute_values")
        pass



    """
        Function for predicting data
    """
    def predict(self, data: dict):
        """function for predicting data

        Args:
            data (dict): data
        """
        print("Ini predict")
        pass

    def predict_recursively(self, data: dict):
        """function for predicting data recursively

        Args:
            data (dict): data
        """
        print("Ini predict_recursively")
        pass



    """
        Function for printing the tree
    """
    def print_tree(self):
        """function for printing the tree"""
        print("Ini print_tree")
        pass

    def print_tree_recursively(self):
        """function for printing the tree recursively"""
        print("Ini print_tree_recursively")
        pass



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
        print("Ini save_tree")
        pass

if __name__ == "__main__":
    example_data1 = {
        "Day" : ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14"],
        "Outlook" : ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
        # "Temperature" : [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
        "Temperature" : ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
        "Humidity" : ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
        "Wind" : ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
        "PlayTennis" : ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
    }

    example_data2 = {
        "Day" : ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14"],
        "Outlook" : ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
        # "Temperature" : [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
        "Temperature" : ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool", "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
        "Humidity" : ["High", "High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
        "Wind" : ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
        "PlayTennis" : ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"]
    }

    df1 = pd.DataFrame(example_data1)
    df2 = pd.DataFrame(example_data2)
    # print(df1)
    # print(df2)

    id3 = ID3()

    """
        Unit Test
    """
    # test plurality_value
    print("======================= Test plurality_value =======================")
    print(f'Plurality Value Outlook: {id3.plurality_value(df1, "Outlook")}')          # Sunny
    print(f'Plurality Value Temperature: {id3.plurality_value(df1, "Temperature")}')  # Mild
    print(f'Plurality Value Humidity: {id3.plurality_value(df1, "Humidity")}')        # High
    print(f'Plurality Value Wind: {id3.plurality_value(df1, "Wind")}')                # Weak
    print(f'Plurality Value PlayTennis: {id3.plurality_value(df1, "PlayTennis")}')    # Yes


    # test is_homogeneous
    print("======================= Test is_homogeneous =======================")
    print(f'Is Homogeneous df1: {id3.is_homogeneous(df1)}')  # False
    print(f'Is Homogeneous df2: {id3.is_homogeneous(df2)}')  # True


    # test entropy
    print("======================= Test entropy =======================")
    # entropy for all data
    print(f'Entropy df1: {id3.entropy(df1)}')  # 0.940286
    print(f'Entropy df2: {id3.entropy(df2)}')  # 0.0

    # entropy for Outlook = Sunny
    df_sunny = df1[df1["Outlook"] == "Sunny"]
    print(f'Entropy df_sunny: {id3.entropy(df_sunny)}')  # 0.970951

    # entropy for Outlook = Overcast
    df_overcast = df1[df1["Outlook"] == "Overcast"]
    print(f'Entropy df_overcast: {id3.entropy(df_overcast)}')  # 0.0

    # entropy for Outlook = Rain
    df_rain = df1[df1["Outlook"] == "Rain"]
    print(f'Entropy df_rain: {id3.entropy(df_rain)}')  # 0.970951


    # test information_gain
    print("======================= Test information_gain =======================")
    print(f'Information Gain Outlook: {id3.information_gain(df1, "Outlook")}')          # 0.246749
    print(f'Information Gain Temperature: {id3.information_gain(df1, "Temperature")}')  # 0.029223
    print(f'Information Gain Humidity: {id3.information_gain(df1, "Humidity")}')        # 0.151835
    print(f'Information Gain Wind: {id3.information_gain(df1, "Wind")}')                # 0.048127

    # test find_best_attribute
    print("======================= Test find_best_attribute =======================")
    print(f'Best Attribute: {id3.find_best_attribute(df1, ["Outlook", "Temperature", "Humidity", "Wind"])}')  # Outlook

    # test break_point
    print("======================= Test break_point =======================")
    data3 = {
        "Temperature2": [40, 48, 60, 72, 80, 90],
        "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No"]
    }
    df3 = pd.DataFrame(data3)
    print(f'Best Break Point: {id3.break_point(df3, "Temperature2")}')
    # Best Break Point: 54.0, Gain: 0.459148
    # Break Point: 54.0, Gain: 0.459148
    # Break Point: 85.0, Gain: 0.190875

