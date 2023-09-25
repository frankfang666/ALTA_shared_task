import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import RobertaTokenizer, RobertaModel
from Datasets import TrainSet, ValidationSet
from LoadData import read_json_objs
from TextClassifier import TextClassifier
from torch.utils.data import DataLoader

from TrainPredict import train, create_output_file, predict
from Arguments import args

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

roberta_base_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
roberta_base_model = RobertaModel.from_pretrained('roberta-base')

roberta_large_tokenizer = RobertaTokenizer.from_pretrained('roberta-large', truncation=True, do_lower_case=True)
roberta_large_model = RobertaModel.from_pretrained('roberta-large')

deberta_base_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', use_fast=True)
deberta_base_model = AutoModel.from_pretrained('microsoft/deberta-v3-base')

deberta_large_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large', use_fast=True)
deberta_large_model = AutoModel.from_pretrained('microsoft/deberta-v3-large')

if args.model == 'roberta_base':
    model = roberta_base_model
    tokenizer = roberta_base_tokenizer
elif args.model == 'roberta_large':
    model = roberta_large_model
    tokenizer = roberta_large_tokenizer
elif args.model == 'deberta_base':
    model = deberta_base_model
    tokenizer = deberta_base_tokenizer
elif args.model == 'deberta_large':
    model = deberta_large_model
    tokenizer = deberta_large_tokenizer
else:
    raise Exception('Invalid model')

path = args.datapath
train_objs, val_objs = read_json_objs(path + 'training.json'), read_json_objs(path + 'validation_data.json')

print("Creating the classifier, initialised with pretrained parameters...")
net = TextClassifier(model)
net.to(device) #Enable gpu support for the model
print("Done creating the classifier.")

criterion = nn.BCEWithLogitsLoss()
opti = optim.AdamW(net.parameters(), lr = 2e-5)

#Creating instances of training and development dataloaders
train_data = TrainSet(train_objs, tokenizer, 100)
val_data = ValidationSet(val_objs, tokenizer, 100)

train_loader = DataLoader(train_data, batch_size = 64)
dev_loader = DataLoader(val_data, batch_size = 64)

print("Done preprocessing training and development data.")

#fine-tune the model
num_epoch = 5

if __name__ == '__main__':
    train(net, device, criterion, opti, train_loader, num_epoch)
    create_output_file(predict(net, dev_loader, device))
