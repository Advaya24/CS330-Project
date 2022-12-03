"""Utilities for scoring the model."""
import torch
import numpy as np


def score(logits, labels, print_=False):
    """Returns the mean accuracy of a model's predictions on a set of examples.

    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples,)
    """

    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]
    # if print_:
    #     for label in np.unique(labels.cpu().numpy())[-3:]:
    #         print(f"{label}: {logits[labels==label].max(dim=-1)[0].mean()}")
    y = torch.argmax(logits, dim=-1) == labels
    y = y.type(torch.float)
    if print_:
        for label in np.unique(labels.cpu().numpy()):
            print(f"{label}: {y[labels==label].mean().item()}")
    return torch.mean(y).item()

def bin_score(prob, labels, print_=False):
    assert prob.dim() == 1
    assert labels.dim() == 1
    assert prob.shape[0] == labels.shape[0]
    y = torch.round(prob) == labels
    y = y.type(torch.float)
    if print_:
        for label in np.unique(labels.cpu().numpy()):
            print(f"{label}: {y[labels==label].mean().item()}")
    return torch.mean(y).item()