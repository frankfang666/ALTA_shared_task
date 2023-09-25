import argparse

myParser = argparse.ArgumentParser(description='Train the model and produce the output file')
myParser.add_argument('-m', '--model', choices=['roberta_base', 'roberta_large', 'deberta_base', 'deberta_large'], help='The model used for training')
myParser.add_argument('-p', '--datapath', help='The path of the data folder')
myParser.add_argument('-f', '--feature', choices=['stylometric', 'advanced'], help='The incorporated features')

args = myParser.parse_args()
