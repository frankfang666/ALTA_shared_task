import torch
from torch.utils.data import Dataset
from FeatureCopy import FeatureExtration
from Arguments import args

class TrainSet(Dataset):
    def __init__(self, objs, tokenizer, maxlen):
        self.data = objs
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]['text'], self.data[index]['label']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.maxlen,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attn_masks = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attn_masks': torch.tensor(attn_masks, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(label, dtype=torch.float)
        } if args.feature is None else {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attn_masks': torch.tensor(attn_masks, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(label, dtype=torch.float),
            'features': FeatureExtration(text)
        }


class ValidationSet(Dataset):
    def __init__(self, objs, tokenizer, maxlen):
        self.data = objs
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]['text']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.maxlen,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attn_masks = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attn_masks': torch.tensor(attn_masks, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        } if args.feature is None else {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attn_masks': torch.tensor(attn_masks, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'features': FeatureExtration(text)
        }

