import torch
import torch.nn as nn
from Arguments import args
import time

class FeatureClassifier(nn.Module):

    def __init__(self):
        super(TextClassifier, self).__init__()

        #Classification layer
        self.cls_layer = nn.Linear(20, 1)

    def forward(self, features):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        logits = self.cls_layer(features)

        return logits


def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc


def evaluate_features(net, device, criterion, dataloader):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for data in dataloader:
            features = data['features'].to(device)
            logits = net(features)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count


def train_features(net, device, criterion, opti, train_loader, max_eps):

    best_f1 = 0
    st = time.time()
    for ep in range(max_eps):

        net.train()

        for it, data in enumerate(train_loader):

            #Clear gradients
            opti.zero_grad()

            features = data['features'].to(device)
            
            #Obtaining the logits from the model
            logits = net(features)

            #Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())

            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            opti.step()

            if it % 100 == 0:

                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss: {}; Accuracy: {}; Time taken (s): {}".format(it, ep, loss.item(), acc, (time.time()-st)))
                st = time.time()
    torch.save(net, './sstcls.pt')


def predict_features(net, dataloader, device):
    net.eval()

    predictions = []

    with torch.no_grad():
        for data in dataloader:
            features = data['features'].to(device)
            logits = net(features)
            probs = torch.sigmoid(logits.unsqueeze(-1))
            soft_probs = (probs > 0.5).long()
            soft_probs.squeeze()
            predictions += torch.flatten(soft_probs).tolist()
    return predictions