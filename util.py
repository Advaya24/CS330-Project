"""Utilities for scoring the model."""
import torch
import numpy as np

eps_scores = {0: [], 1:[]}

# class_accs = {i: {} for i in range(8)}
y_trues, y_preds = [], []

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
    y_trues.append(torch.argmax(logits, dim=-1))
    y_preds.append(labels)
    y = torch.argmax(logits, dim=-1) == labels
    y = y.type(torch.float)
    if print_:
        for label in np.unique(labels.cpu().numpy()):
            print(f"{label}: {y[labels==label].mean().item()}")
            # for label_ in np.unique(labels.cpu().numpy()):
            #     class_accs[label][label_] += 
    return torch.mean(y).item()

def bin_score(prob, labels, print_=False):
    assert prob.dim() == 1
    assert labels.dim() == 1
    assert prob.shape[0] == labels.shape[0]
    y = torch.round(prob) == labels
    y = y.type(torch.float)
    if print_:
    #     print(len(y))
        for label in np.unique(labels.cpu().numpy()):
    #         print(f"{label}: {y[labels==label].mean().item()}")
            eps_scores[label].append((y[labels==label].sum().item(), len(y[labels==label])))
    return torch.mean(y).item()

def return_eps_confusion(dict_confusion, key):
    confusion = {0: {0: 0, 1:0}, 1: {0: 0, 1:0}}
    for label in confusion:
        num_correct = 0
        total = 0
        for corrects, counts in eps_scores[label]:
            num_correct += corrects
            total += counts
        confusion[label][label] = int(num_correct)
        confusion[label][1-label] = int(total - num_correct)
        eps_scores[label] = []
    dict_confusion[key] = confusion
    # print(dict_confusion)
    return dict_confusion

def get_preds_and_labels():
    global y_preds, y_trues
    out_preds, out_trues = torch.cat(y_preds).cpu().numpy(), torch.cat(y_trues).cpu().numpy()
    y_preds, y_trues = [], []
    return out_preds, out_trues