import json
import time
import torch

def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc


def evaluate(net, device, criterion, dataloader):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for data in dataloader:
            text, input_ids, attn_masks, token_type_ids, labels = data['text'].to(device), data['input_ids'].to(device), data['attn_masks'].to(device), data['token_type_ids'].to(device), data['targets'].to(device)
            logits = net(input_ids, attn_masks, token_type_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count


def train(net, device, criterion, opti, train_loader, max_eps):

    best_f1 = 0
    st = time.time()
    for ep in range(max_eps):

        net.train()

        for it, data in enumerate(train_loader):

            #Clear gradients
            opti.zero_grad()

            input_ids, attn_masks, token_type_ids, labels = data['text'].to(device), data['input_ids'].to(device, dtype = torch.long), data['attn_masks'].to(device, dtype = torch.long), data['token_type_ids'].to(device, dtype = torch.long), data['targets'].to(device, dtype = torch.long)

            #Obtaining the logits from the model
            logits = net(input_ids, attn_masks, token_type_ids)

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


def predict(net, dataloader, device):
    net.eval()

    predictions = []

    with torch.no_grad():
        for data in dataloader:
            input_ids, attn_masks, token_type_ids = data['text'].to(device), data['input_ids'].to(device), data['attn_masks'].to(device), data['token_type_ids'].to(device)
            logits = net(input_ids, attn_masks, token_type_ids)
            probs = torch.sigmoid(logits.unsqueeze(-1))
            soft_probs = (probs > 0.5).long()
            soft_probs.squeeze()
            predictions += torch.flatten(soft_probs).tolist()
    return predictions


def create_output_file(predictions):
  with open('./answer.json', 'w+') as f:
    for i in range(len(predictions)):
      f.write(json.dumps({'id': i, 'label': predictions[i]})+'\n')