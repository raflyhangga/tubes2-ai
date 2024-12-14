class Node:
    def __init__(self, attribute: str, value: str|int|float|bool, children: dict):
        self.attribute = attribute
        self.value = value
        self.children = children

class ID3:
    def __init__(self):
        print("Ini konstruktor")
        pass
    
    """ 
        Function for training ID3 model 
    """
    def train(self, data: dict, attribute: str, parent_data: dict) -> None:
        """function for training ID3 model

        Args:
            data (dict): data for current node
            attribute (str): the remaining attribute
            parent_data (dict): data from parent node
        """
        print("Ini train")
        pass

    def plurality_value(self, data: dict) -> str|int|float|bool|None:
        """function for finding the most common value in data

        Args:
            data (dict): data

        Returns:
            str|int|float|bool|None: the most common value in data
        """
        print("Ini plurality_value")
        pass

    def is_homogeneous(self, data: dict) -> bool:
        """function for checking if data is homogeneous

        Args:
            data (dict): data

        Returns:
            bool: True if data is homogeneous, otherwise False
        """
        print("Ini is_homogeneous")
        pass

    def entropy(self, data: dict) -> float:
        """function for calculating entropy

        Args:
            data (dict): data

        Returns:
            float: entropy
        """
        print("Ini entropy")
        pass

    def information_gain(self, data: dict, attribute: str) -> float:
        """function for calculating information gain

        Args:
            data (dict): data
            attribute (str): attribute to be calculated

        Returns:
            float: information gain
        """
        print("Ini information_gain")
        pass

    def find_best_attribute(self, data: dict, attribute: str) -> str:
        """function for finding the best attribute

        Args:
            data (dict): data
            attribute (str): the remaining attribute

        Returns:
            str: the best attribute
        """
        print("Ini find_best_attribute")
        pass

    def split_data(self, data: dict, attribute: str) -> dict:
        """function for splitting data based on attribute

        Args:
            data (dict): data
            attribute (str): attribute
            value (str|int|float|bool): value of the attribute

        Returns:
            dict: splitted data
        """
        print("Ini split_data")
        pass

    def filter_data(self, data: dict, attribute: str, value: str|int|float|bool) -> dict:
        """function for filtering data based on attribute and value

        Args:
            data (dict): data
            attribute (str): attribute
            value (str|int|float|bool): value of the attribute

        Returns:
            dict: filtered data
        """
        print("Ini filter_data")
        pass

    def break_point(self, data: dict, attribute: str) -> dict:
        """function for finding break point of continuous data

        Args:
            data (dict): data
            attribute (str): attribute

        Returns:
            dict: break point
        """
        print("Ini break_point")
        pass

    def get_attribute_type(self, data: dict, attribute: str) -> str:
        """function for getting attribute type

        Args:
            data (dict): data
            attribute (str): attribute

        Returns:
            str: attribute type
        """
        print("Ini get_attribute_type")
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