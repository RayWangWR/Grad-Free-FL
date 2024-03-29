import torch
import torch.nn as nn
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import _get_func_signature
from sklearn.metrics import f1_score, accuracy_score
from transformers import RobertaTokenizer
from utils import hinge_loss

class Metric(MetricBase):
    def __init__(self, pred=None, target=None, seq_len=None, tokenizer=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self._pred = []
        self._target = []
        self.hinge = 0.0
        self.ce_loss = 0.0
        self.ce_fct = nn.CrossEntropyLoss(reduction='sum')
        self.margin = 2
        if tokenizer is None:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.label_map = {
            tokenizer.encode('bad', add_special_tokens=False)[0]: 0,  # negative
            tokenizer.encode('great', add_special_tokens=False)[0]: 1,  # positive
        }

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")
        # pred: batch_size x seq_len x vocab_size
        self.ce_loss += self.ce_fct(pred, target).item()

        # calculate hinge loss
        hinge_target = target.clone()
        for key, val in self.label_map.items():
            hinge_target[target==key] = val

        for t in hinge_target.cpu().numpy().tolist():
            self._target.append(t)

        interest_index = list(self.label_map.keys())
        pred = pred[:, interest_index]
        self.hinge += hinge_loss(pred, hinge_target, self.margin, reduction='sum').item()
        
        pred = pred.argmax(dim=-1).detach().cpu().numpy().tolist()
        self._pred.extend(pred)


    def get_metric(self, reset=True):
        acc = accuracy_score(self._target, self._pred)
        hinge_loss = self.hinge / len(self._target)
        ce_loss = self.ce_loss / len(self._target)
        if reset:
            self._target = []
            self._pred = []
            self.hinge = 0.0
            self.ce_loss = 0.0
        return {'acc': acc,
                'hinge': hinge_loss,
                'ce': ce_loss}

