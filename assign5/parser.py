import pandas as pd

class Parser(object):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, header=None)
    
    def _getAtrributes(self):
        class_index = self.df.iloc[:,0].values
        title = self.df.iloc[:,1].values
        description = self.df.iloc[:,2].values
        return (class_index, title, description)

from argparse import ArgumentParser
class CLI_Parser(object):
    def __init__(self):
        arguments = {
            '--path': {'type': str, 'required': True}
        }
        parser = ArgumentParser()
        for arg_name, attr in arguments.items(): 
            parser.add_argument(arg_name, type=attr['type'], required=attr['required'])
        
        self.args = parser.parse_args()
    
    # return each arguments values
    def __call__(self):
        return tuple(vars(self.args).values())

        

def main():
    cli_parse = CLI_Parser()
    args = cli_parse()
    parser = Parser(args[0])
    print(parser._getAtrributes()[2][-2])

if __name__ == "__main__":
    main()


    