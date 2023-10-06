import sklearn
from sklearn.ensemble import AdaBoostClassifier
from FeatureCopy import FeatureExtration
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

def concat_features(model, dataset, is_train, device):
    input_ids = torch.stack([data['input_ids'] for data in dataset]).to(device)
    attention_masks = torch.stack([data['attn_masks'] for data in dataset]).to(device)
    token_type_ids = torch.stack([data['token_type_ids'] for data in dataset]).to(device)
    embeddings = model(input_ids, attention_masks, token_type_ids)[1].numpy()
    features = np.array([FeatureExtration(data['text']) for data in dataset])
    normalised_features = normalize(features, axis=0, norm='l1')
    if is_train:
        return np.concatenate((embeddings, normalised_features), axis=1), [data['target'].numpy() for data in dataset]
    else:
        return np.concatenate((embeddings, normalised_features), axis=1)

def train_features(features, targets):
    classifier = LogisticRegression()
    classifier.fit(features, targets)
    return classifier
    