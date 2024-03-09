from collections import defaultdict
import torch
from torch import nn
import random


class RunningMean:
    def __init__(self):
        self.cnt = 0
        self.total = 0
    
    def get(self):
        return self.total / (1e-9 + self.cnt)
    
    def add(self, value):
        self.total += value
        self.cnt += 1
    
    def reset(self):
        self.cnt = 0
        self.total = 0


class MetricTracker:
    def __init__(self):
        self.metrics = defaultdict(RunningMean)

    def at(self, key):
        return self.metrics[key].get()
    
    def reset(self):
        for key in self.metrics:
            self.metrics[key].reset()
    
    def add(self, key, value):
        self.metrics[key].add(value)
    
    def to_dict(self):
        out = {}
        for key in self.metrics:
            out[key] = self.metrics[key].get()
        return out


def calc_grad_norm(params):
    with torch.no_grad():
        total = 0
        for p in params:
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

def get_true_domains(attributes):
    # ToDo: rewrite if needed
    true_domains = []
    attributes = attributes.detach().cpu()
    for i in range(attributes.shape[0]):
        good_idx = [j for j in range(attributes.shape[1]) if attributes[i][j] == 1]
        true_domains.append(random.choice(good_idx))

    return torch.tensor(true_domains, device=attributes.device)


def apply_label_smoothing(labels, k=0.0):
    C = 100
    sign = 1
    if (labels - torch.ones_like(labels)).abs().sum() <= 1e-5:
        sign = -1
    return labels - sign * k * torch.randint(low=0, high=2 * C, 
                                             size=labels.shape, device=labels.device) / C


def double(t):
    repeat = [1] * len(t.shape)
    repeat[0] = 2
    return t.repeat(repeat)


class SuperDict:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._keys = list(kwargs.keys())

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self._keys.add(key)
        return setattr(self, key, value)

    def apply(self, f):
        for key in self._keys:
            f(self[key])

    def applym(self, s, *args):
        for key in self._keys:
            getattr(self[key], s)(*args)


def ema(ema_net, net, beta=0.999):
    ema_state_dict = ema_net.state_dict()
    for key, value in net.state_dict().items():
        ema_state_dict[key] = ema_state_dict[key] * beta + (1 - beta) * value
    ema_net.load_state_dict(ema_state_dict)


# Transformations to be applied to each individual image sample
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
