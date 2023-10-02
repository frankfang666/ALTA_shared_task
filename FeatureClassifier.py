import sklearn
from sklearn.ensemble import AdaBoostClassifier
from FeatureCopy import FeatureExtration
import numpy as np
import torch

def concat_features(model, data):
    model()