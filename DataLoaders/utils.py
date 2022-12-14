import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np

def get_sampler(labels,class_weights):
    example_weights = [class_weights[e] for e in labels]
    sampler = WeightedRandomSampler(example_weights,len(labels))   
    return sampler

def get_class_weights(labels):
    labels_unique, counts = np.unique(labels,return_counts=True)
    print(f"Unique labels:{labels_unique}. Counts:{counts}")
    class_weights = [sum(counts)/c for c in counts]
    return class_weights
