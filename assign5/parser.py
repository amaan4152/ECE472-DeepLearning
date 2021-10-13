import pandas as pd

class Parser(object):
    def __init__(self, train_path, test_path):
        self.df_train = pd.read_csv(train_path, header=None)
        self.df_test = pd.read_csv(test_path)

    def _getAtrributes(self):
        train_class_index = self.df_train.iloc[:,0].values
        train_title = self.df_train.iloc[:,1].values
        train_description = self.df_train.iloc[:,2].values

        test_class_index = self.df_test.iloc[:,0].values
        test_title = self.df_test.iloc[:,1].values
        test_description = self.df_test.iloc[:,2].values

        return ((train_class_index, train_title, train_description),
               (test_class_index, test_title, test_description))

from argparse import ArgumentParser
class CLI_Parser(object):
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument('--train', type=str, required=True, help="provide train data path")
        parser.add_argument('--test', type=str, required=True, help="provide test data path")
        self.args = parser.parse_args()
    
    # return each arguments values
    def __call__(self):        
        return self.args

    