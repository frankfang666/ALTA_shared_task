import argparse

myParser = argparse.ArgumentParser(description='Train the model and produce the output file')
myParser.add_argument('-m', '--model', choices=['bert_base', 'bert_large', 'roberta_base', 'roberta_large', 'deberta_base', 'deberta_large'], help='The model used for training')
myParser.add_argument('-p', '--datapath', help='The path of the data folder')
myParser.add_argument('-l', '--load', help='Load the existing model')
myParser.add_argument('-e', '--epochs', type=int, default=5, help='The training epochs')
myParser.add_argument('-x', '--maxlen', type=int, default=100, help='Max numbers of tokens')
myParser.add_argument('-b', '--batchsize', type=int, default=64, help='Training batch size')

args = myParser.parse_args()
