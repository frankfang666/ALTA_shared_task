import json

import torch
from torch.utils.data import Dataset
import random

from Training import roberta_tokenizer

machine_path = './test_data/GPT-3/'
gpt_objs_1 = list(json.load(open(machine_path + 'data.json')))
gpt_objs_2 = list(json.load(open(machine_path + 'data_2.json')))
gpt_objs = gpt_objs_1 + gpt_objs_2
# print(len(gpt_objs))

human_path = './test_data/Human/'
human_objs_1 = list(json.load(open(human_path + 'data.json')))
human_objs_2 = list(json.load(open(human_path + 'data_2.json')))
human_objs = human_objs_1 + human_objs_2
# print(len(human_objs))

class DS(Dataset):

    def __init__(self, objs, maxlen, tokenizer, label):


        #Initialize the BERT tokenizer
        self.tokenizer = tokenizer
        self.data = objs
        self.label = label
        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        inputs = self.tokenizer.encode_plus(
            self.data[index]['document'],
            None,
            add_special_tokens=True,
            max_length=self.maxlen,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        tokens_ids_tensor = inputs['input_ids']
        attn_mask = inputs['attention_mask']

        return torch.tensor(tokens_ids_tensor, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long), torch.tensor(self.label, dtype=torch.long)


random.shuffle(gpt_objs)
random.shuffle(human_objs)

gpt_train_size = int(0.8 * len(gpt_objs))
human_train_size = int(0.8 * len(human_objs))
gpt_objs_train = gpt_objs[:gpt_train_size]
gpt_objs_test = gpt_objs[gpt_train_size:]
human_objs_train = human_objs[:human_train_size]
human_objs_test = human_objs[human_train_size:]

maxlen = 300
gpt_train_set = DS(gpt_objs_train, maxlen, roberta_tokenizer, 0)
human_train_set = DS(human_objs_train, maxlen, roberta_tokenizer, 0)
gpt_test_set = DS(gpt_objs_test, maxlen, roberta_tokenizer, 0)
human_test_set = DS(human_objs_test, maxlen, roberta_tokenizer, 0)

train_set = torch.utils.data.ConcatDataset([gpt_train_set, human_train_set])
test_set = torch.utils.data.ConcatDataset([gpt_test_set, human_test_set])
