import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from Datasets import TrainSet, ValidationSet
from LoadData import read_json_objs
from TextClassifier import TextClassifier
from torch.utils.data import DataLoader

from TrainPredict import train, create_output_file, predict
from Arguments import args

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


if args.model == 'bert_base':
    model = BertModel.from_pretrained('bert-base-cased')
    tokenizer =  BertTokenizer.from_pretrained('bert-base-cased')
elif args.model == 'bert_large':
    model = BertModel.from_pretrained('bert-large-cased')
    tokenizer =  BertTokenizer.from_pretrained('bert-large-cased')
elif args.model == 'roberta_base':
    model = RobertaModel.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
elif args.model == 'roberta_large':
    model = RobertaModel.from_pretrained('roberta-large')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large', truncation=True, do_lower_case=True)
elif args.model == 'deberta_base':
    model =  AutoModel.from_pretrained('microsoft/deberta-v3-base')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', use_fast=True)
elif args.model == 'deberta_large':
    model = AutoModel.from_pretrained('microsoft/deberta-v3-large')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large', use_fast=True)
else:
    raise Exception('Invalid model')

path = args.datapath
train_objs, val_objs, test_objs = read_json_objs(path + 'training.json'), read_json_objs(path + 'validation_data.json'), read_json_objs(path + 'test_data.json')

print("Creating the classifier, initialised with pretrained parameters...")
net = TextClassifier(model)
net.to(device) #Enable gpu support for the model
print("Done creating the classifier.")

criterion = nn.BCEWithLogitsLoss()
opti = optim.AdamW(net.parameters(), lr = 2e-5)

#Creating instances of training and development dataloaders
train_data = TrainSet(train_objs, tokenizer, args.maxlen)
val_data = ValidationSet(val_objs, tokenizer, args.maxlen)
test_data = ValidationSet(test_objs, tokenizer, args.maxlen)

train_loader = DataLoader(train_data, batch_size = args.batchsize)
dev_loader = DataLoader(val_data, batch_size = args.batchsize)
test_loader = DataLoader(test_data, batch_size = args.batchsize)

print("Done preprocessing training and development data.")

#fine-tune the model
num_epoch = args.epochs

if __name__ == '__main__':
    neural_model = None
    if args.load is None:
        train(net, device, criterion, opti, train_loader, num_epoch)
        neural_model = net
    else:
        neural_model = torch.load(args.load)
    create_output_file(predict(neural_model, test_loader, device))
