import math

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def eval_accuracy(all_label_probs, test_labels, mode='diagonal_W', p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    probs = []
    assert len(all_label_probs) == len(test_labels)
    for label_probs, true_label in zip(all_label_probs, test_labels):
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
        calibrate_label_probs /= np.sum(calibrate_label_probs)
        probs.append(calibrate_label_probs)

        ans_label = np.argmax(calibrate_label_probs)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list), probs

LABEL_DICT = {0: ['yes',], 1: ['no',]}
LABEL_TO_INT = {'yes': 0, 'no': 1}

def calibrate_label_dict(logits, tokenizer, label_dict=LABEL_DICT, top_k=10, apply_softmax=True, content_free_inputs=('N/A',), return_logits = False):
    probs = logits.float().cpu() if not apply_softmax else torch.softmax(logits, dim=-1).float().cpu()
    top_probs, top_tokens = torch.topk(probs, k=top_k)
    temp = {}
    scores = {}
    for prob, token in zip(top_probs[0], top_tokens[0]):
        str_token = tokenizer.decode(token.item())
        str_token = str_token.lower().strip()
        if str_token not in temp.keys():
            temp[str_token] = prob.item()
            scores[str_token] = logits[0][token].item()
        else:
            pass
    if return_logits:
        return temp, scores
    return temp

def get_prob_from_logits(top_token_probs, label_dict=LABEL_DICT):
    p_y = [0] * len(label_dict)
    for i, answers in label_dict.items():
        prob = 0
        for a in answers:
            a = a.lower()
            if a not in top_token_probs.keys():
                prob += 0
            else:
                prob += top_token_probs[a]
        p_y[i] = prob
    return p_y