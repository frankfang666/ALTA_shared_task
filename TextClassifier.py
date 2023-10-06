import torch
import torch.nn as nn
from Arguments import args

class TextClassifier(nn.Module):

    def __init__(self, model):
        super(TextClassifier, self).__init__()
        #Instantiating BERT model object
        self.pretrained_layer = model

        #Classification layer
        if args.feature != 'concat':
            self.cls_layer = nn.Linear(768, 1) if 'base' in args.model else nn.Linear(1024, 1)
        else:
            self.cls_layer = nn.Linear(788, 1) if 'base' in args.model else nn.Linear(1044, 1)

    def forward(self, input_ids, attn_masks, token_type_ids, features=None):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        outputs = self.pretrained_layer(input_ids=input_ids, attention_mask = attn_masks, token_type_ids=token_type_ids, return_dict=True)
        cont_reps = outputs.last_hidden_state

        #Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]

        #Feeding cls_rep to the classifier layer
        if args.features != 'concat':
            logits = self.cls_layer(cls_rep)
        else:
            cat_rep = torch.cat((cls_rep, features), dim=1)
            logits = self.cls_layer(cat_rep)

        return logits, cls_rep
