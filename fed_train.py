import os
import copy
import time
import math
import random

import torch
# import fitlog
import argparse
import numpy as np
import cma
from fastNLP import cache_results, Tester, DataSet
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    BertConfig,
    BertTokenizer,
    ElectraConfig,
    ElectraTokenizer,
    BartConfig,
    BartTokenizer,
    T5Config,
    T5Tokenizer,
    GPT2Config,
    GPT2Tokenizer,
    BartConfig as CPTConfig,
)
from models.modeling_roberta import RobertaForMaskedLM
from utils import hinge_loss
from sklearn.metrics import f1_score

import copy
from sklearn import linear_model
from dataloader import sent140Loader
from metrics import Metric

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='roberta-large', type=str)
parser.add_argument("--task_name", default='sent140', type=str)
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--intrinsic_dim", default=500, type=int)
parser.add_argument("--k_shot", default=16, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--budget", default=8000, type=int)
parser.add_argument("--popsize", default=20, type=int)
parser.add_argument("--bound", default=0, type=int)
parser.add_argument("--sigma", default=1, type=float)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--eval_every", default=100, type=int)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--alg", default='CMA', type=str)
parser.add_argument("--random_proj", default='normal', type=str)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--loss_type", default='ce', type=str)
parser.add_argument("--cat_or_add", default='add', type=str)
parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')
parser.add_argument(
    "--inference_framework",
    default='pt',
    type=str,
    help='''Which inference framework to use. 
         Currently supports `pt` and `ort`, standing for pytorch and Microsoft onnxruntime respectively'''
)
parser.add_argument(
    "--onnx_model_path",
    default=None,
    type=str,
    help='Path to your onnx model.'
)
args = parser.parse_args()

# task_name = args.task_name
n_prompt_tokens = args.n_prompt_tokens
intrinsic_dim = args.intrinsic_dim
k_shot = args.k_shot
batch_size = args.batch_size
budget = args.budget
bound = args.bound
sigma = args.sigma
# bound = math.sqrt(intrinsic_dim)
# if random_proj == 'normal':
#     bound = math.pow(intrinsic_dim, 0.75)
# elif model_name in ['t5-small', 't5-base', 't5-large', 't5-3b']:
#     bound = 100
# else:
#     bound = 5
if args.popsize > 0:
    popsize = args.popsize
else:
    popsize = 4 + 3 * np.log(intrinsic_dim)
device = args.device
alg = args.alg
random_proj = args.random_proj
seed = args.seed
loss_type = args.loss_type
print_every = args.print_every
eval_every = args.eval_every
# if task_name in ['mrpc', 'snli', 'qnli', 'rte']:
#     args.cat_or_add = 'cat'
cat_or_add = args.cat_or_add
parallel = args.parallel
inference_framework = args.inference_framework
onnx_model_path = args.onnx_model_path

if inference_framework not in ['pt', 'ort']:
    raise ValueError(f'inference_framework only supports "pt", "ort", got `{inference_framework}` instead.')
if inference_framework == 'ort':
    assert onnx_model_path is not None, 'Path to onnx model is required, got None instead.'
    assert os.path.exists(onnx_model_path), f'In valid onnx model path `{onnx_model_path}`'



args.bbt_version = 'bbt'


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class LMForwardAPI:
    def __init__(self, model_name='roberta-large', n_prompt_tokens=50, task_name='sent140',
                 loss_type='hinge', init_prompt_path=None):
        if True:
            self.config = RobertaConfig.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForMaskedLM.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
                inference_framework=inference_framework,
                onnx_model_path=onnx_model_path,
            )
            self.model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(self.config.vocab_size))

        if inference_framework == 'ort':
            self.model.roberta = None
        if cat_or_add == 'cat':
            self.model.set_concat_prompt(True)
            if init_prompt_path is not None:
                print('Initialize prompt embedding from {}'.format(init_prompt_path))
                self.init_prompt = torch.load(init_prompt_path).weight.cpu().reshape(-1)
            else:
                print('Initial prompt embedding not found. Initialize to zero embedding.')
                self.init_prompt = torch.zeros(n_prompt_tokens * self.config.hidden_size)
            print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))
        else:
            # self.model.set_concat_prompt(False)
            self.init_prompt = None
        self.model.to(device)
        self.model.eval()
        self.linear = torch.nn.Linear(intrinsic_dim, n_prompt_tokens * self.config.hidden_size, bias=False)
        if random_proj == 'normal':
            # calculate std for normal distribution
            if True:
                embedding = self.model.roberta.get_input_embeddings().weight.clone().cpu()
            mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = std_hat / (np.sqrt(intrinsic_dim) * args.sigma)
            # temp = intrinsic_dim - std_hat * std_hat
            # mu = mu_hat / temp
            # std = std_hat / np.sqrt(temp)
            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear.parameters():
                torch.nn.init.normal_(p, mu, std)
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_prompt = None
        self.num_call = 0
        # self.save_path = save_path
        self.print_every = print_every
        self.eval_every = eval_every
        self.loss_type = loss_type
        # if save_path is not None:
        #     os.makedirs(save_path, exist_ok=True)
        self.metric = {}
        self.metric_key = {}
        self.metric_name = {}
        
        
        self.metric['sent140'] = Metric(target='labels', pred='logits', tokenizer=tokenizer)
        self.metric_key['sent140'] = 'acc'
        self.metric_name['sent140'] = 'BinaryClassificationMetric'
        
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

#     def calc_metric(self, logits, target):
#         label_map = self.metric.label_map

#         converted_target = target.clone()
#         for key, val in label_map.items():
#             converted_target[target == key] = val
#         interest_index = list(label_map.keys())
#         logits = logits[:, interest_index]
#         pred = logits.argmax(dim=-1)

#         if self.metric_key == 'acc':
#             perf = (pred == converted_target).sum() / len(target)
#         elif self.metric_key == 'f1':
#             perf = f1_score(converted_target.detach().cpu().numpy().tolist(),
#                             pred.detach().cpu().numpy().tolist())
#         else:
#             raise KeyError(f'[Metric] Only support [acc, f1], got {self.metric_key} instead.')

#         if self.loss_type == 'hinge':
#             loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
#         elif self.loss_type == 'ce':
#             loss = self.ce_loss(logits, converted_target).item()
#         elif self.loss_type == 'perf':
#             loss = -1 * perf
#         else:
#             raise KeyError(f'[Loss] Only support [hinge, ce, perf], got {self.loss_type} instead.')

#         return loss, perf


    def calc_metric(self, logits, target, task_name, grad_ce=False):
        label_map = self.metric[task_name].label_map

        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        interest_index = list(label_map.keys())
        logits = logits[:, interest_index]
        pred = logits.argmax(dim=-1)

        if self.metric_key[task_name] == 'acc':
            perf = (pred == converted_target).sum() / len(target)
        elif self.metric_key[task_name] == 'f1':
            perf = f1_score(converted_target.detach().cpu().numpy().tolist(),
                            pred.detach().cpu().numpy().tolist())
        else:
            raise KeyError(f'[Metric] Only support [acc, f1], got {self.metric_key[task_name]} instead.')

        if self.loss_type == 'hinge':
            loss = hinge_loss(logits, converted_target, margin=self.metric[task_name].margin, reduction='sum').item() / len(target)
        elif self.loss_type == 'ce':
            if grad_ce:
                loss = self.ce_loss(logits, converted_target)
            else:
                loss = self.ce_loss(logits, converted_target).item()
        elif self.loss_type == 'perf':
            loss = -1 * perf
        else:
            raise KeyError(f'[Loss] Only support [hinge, ce, perf], got {self.loss_type} instead.')

        return loss, perf
    
    def eval(self, prompt_embedding=None, test_data=None):
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        if test_data is None:
            bsz = len(dev_data['input_ids'])  # batch size of dev data is the orignal batch size of training data
        else:
            bsz = batch_size  # for test data
        tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        if isinstance(prompt_embedding, list):  # multiple queries
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
                if self.init_prompt is not None:
                    z = z + self.init_prompt  # Az + p_0
                pe_list.append(z.reshape(n_prompt_tokens, -1).repeat(bsz, 1, 1))
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim
            assert len(prompt_embedding) == len(train_data['input_ids'])
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(n_prompt_tokens, -1).repeat(bsz, 1, 1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        self.model.set_prompt_embedding(prompt_embedding)

        if isinstance(test_data, DataSet):
            if prompt_embedding.shape[0] > bsz:
                raise ValueError('Provide a single prompt embedding for testing.')
            test_tester = Tester(data=test_data, model=self.model, metrics=self.metric, batch_size=batch_size, num_workers=1, device=device, use_tqdm=True)
            results = test_tester.test()
            test_acc = results[self.metric_name][self.metric_key]
            # fitlog.add_best_metric(test_acc, name='test_acc')
            return test_acc
        else:
            for k, v in train_data.items():
                train_data[k] = v.to(device)
            with torch.no_grad():
                logits = self.model(
                        input_ids=train_data['input_ids'],
                        attention_mask=train_data['attention_mask'],
                        mask_pos=train_data['mask_pos'],
                    )['logits']

            if parallel:  # we have multiple queries
                all_losses, all_perfs = [], []
                for i in range(len(logits) // bsz):
                    tmp_logits = logits[i * bsz:i * bsz + bsz]
                    tmp_target = train_data['labels'][i * bsz:i * bsz + bsz]
                    tmp_loss, tmp_perf = self.calc_metric(tmp_logits, tmp_target)
                    all_losses.append(tmp_loss)
                    all_perfs.append(tmp_perf)
                loss = min(all_losses)
                best_sol = all_losses.index(loss)  # argmin
                perf = all_perfs[best_sol]  # corresponding performance
                tmp_prompt = tmp_prompt[best_sol]  # numpy.ndarray
                prompt_embedding = pe_list[best_sol]  # to be prepended to the input
            else:  # single query
                loss, perf = self.calc_metric(logits, train_data['labels'])
            # fitlog.add_loss(loss, name=self.loss_type, step=self.num_call)
            # fitlog.add_metric(perf, name='train_acc', step=self.num_call)

            if perf > self.best_train_perf:
                self.best_train_perf = perf
                # fitlog.add_best_metric(self.best_train_perf, name='train_acc')

            # if self.save_path is not None:
            #     with open(os.path.join(self.save_path, 'train_acc.txt'), 'a') as fout:
            #         fout.write('{}\t{}\n'.format(self.num_call, perf))

            if self.num_call % self.print_every == 0:
                print(
                    '[# API Calls {}] loss: {}. Current perf: {}. Best perf so far: {}'.format(
                        self.num_call,
                        round(float(loss), 4),
                        round(float(perf), 4),
                        round(float(self.best_train_perf), 4)))

            if self.num_call % self.eval_every == 0:
                print('********* Evaluated on dev set *********')
                if parallel:  # if we have multiple queries, use the one that achieves minimal loss
                    self.model.set_prompt_embedding(prompt_embedding)
                for k, v in dev_data.items():
                    dev_data[k] = v.to(device)
                with torch.no_grad():
                    logits = self.model(
                            input_ids=dev_data['input_ids'],
                            attention_mask=dev_data['attention_mask'],
                            mask_pos=dev_data['mask_pos'],
                        )['logits']

                dev_loss, dev_perf = self.calc_metric(logits, dev_data['labels'])
                # fitlog.add_metric(dev_perf, name='dev_acc', step=self.num_call)
                if dev_perf > self.best_dev_perf:
                    self.best_dev_perf = dev_perf
                    # fitlog.add_best_metric(self.best_dev_perf, name='dev_acc')
                    self.best_prompt = copy.deepcopy(tmp_prompt)
                # if self.save_path is not None:
                #     with open(os.path.join(self.save_path, 'dev_acc.txt'), 'a') as fout:
                #         fout.write('{}\t{}\n'.format(self.num_call, dev_loss))
                print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
                    round(float(dev_loss), 4),
                    round(float(dev_perf), 4),
                    round(float(self.best_dev_perf), 4)))
                print('********* Done *********')
            if parallel:
                return all_losses
            else:
                return loss
            
            
    


    
    def eval_fed(self, task_name, intrincs=None, train_data=None, test_data=None, log_file=None):
        self.num_call += 1
      
        if test_data is None:
            bsz = len(train_data['input_ids'])  # batch size of dev data is the orignal batch size of training data
        else:
            bsz = batch_size  # for test data
#         tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        
        if intrincs is None:
            prompt_embedding = self.init_prompt
        elif isinstance(intrincs, np.ndarray):  # single query or None
            intrincs = torch.tensor(intrincs).type(torch.float32)  # z
            prompt_embedding = self.linear(intrincs)  # Az
            assert self.init_prompt is not None
            prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
#             prompt_embedding = prompt_embedding.reshape(n_prompt_tokens, -1).repeat(bsz, 1, 1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        prompt_embedding = prompt_embedding.reshape(n_prompt_tokens, -1)
        self.model.set_prompt_embedding(prompt_embedding)

        if isinstance(test_data, DataSet):
#             if prompt_embedding.shape[0] > bsz:
#                 raise ValueError('Provide a single prompt embedding for testing.')
            test_tester = Tester(data=test_data, model=self.model, metrics=self.metric[task_name], batch_size=batch_size, num_workers=1, device=device, use_tqdm=True)
            results = test_tester.test()
            test_acc = results[self.metric_name[task_name]][self.metric_key[task_name]]
#             with open(log_file, 'a') as log_f:
#                 print('ce: {}'.format(results[self.metric_name]['ce']), file=log_f)
            # fitlog.add_best_metric(test_acc, name='test_acc')
            return test_acc
        else:
            self.model.eval()
            assert train_data is not None
            for k, v in train_data.items():
                train_data[k] = v.to(device)
            with torch.no_grad():
                    bs = 8
                    logits = []
                    n_samples = train_data['input_ids'].size(0)
                    i = 0
                    while i < n_samples:
                        j = min(n_samples, i+bs)
#                         breakpoint()
                        logits_c = self.model(
                            input_ids=train_data['input_ids'][i:j,:],
                            attention_mask=train_data['attention_mask'][i:j,:],
                            mask_pos=train_data['mask_pos'][i:j],
                            )['logits']
                        logits.append(logits_c)
                        i += bs
                    logits = torch.cat(logits, dim=0)


            if parallel:  # we have multiple queries
                all_losses, all_perfs = [], []
                for i in range(len(logits) // bsz):
                    tmp_logits = logits[i * bsz:i * bsz + bsz]
                    tmp_target = train_data['labels'][i * bsz:i * bsz + bsz]
                    tmp_loss, tmp_perf = self.calc_metric(tmp_logits, tmp_target,)
                    all_losses.append(tmp_loss)
                    all_perfs.append(tmp_perf)
                loss = min(all_losses)
                best_sol = all_losses.index(loss)  # argmin
                perf = all_perfs[best_sol]  # corresponding performance
                tmp_prompt = tmp_prompt[best_sol]  # numpy.ndarray
                prompt_embedding = pe_list[best_sol]  # to be prepended to the input
            else:  # single query
                loss, perf = self.calc_metric(logits, train_data['labels'], task_name=task_name)
#                 loss, perf = self.calc_metric(logits, train_data['labels'], grad_ce=True)
            # fitlog.add_loss(loss, name=self.loss_type, step=self.num_call)
            # fitlog.add_metric(perf, name='train_acc', step=self.num_call)

#             loss.backward()
            
            if perf > self.best_train_perf:
                self.best_train_perf = perf
                # fitlog.add_best_metric(self.best_train_perf, name='train_acc')

            # if self.save_path is not None:
            #     with open(os.path.join(self.save_path, 'train_acc.txt'), 'a') as fout:
            #         fout.write('{}\t{}\n'.format(self.num_call, perf))

            if self.num_call % self.print_every == 0:
                print(
                    '[# API Calls {}] loss: {}. Current perf: {}. Best perf so far: {}'.format(
                        self.num_call,
                        round(float(loss), 4),
                        round(float(perf), 4),
                        round(float(self.best_train_perf), 4)))
                if log_file is not None:
                    with open(log_file, 'a') as log_f:
                        print(
                    '[# API Calls {}] loss: {}. Current perf: {}. Best perf so far: {}'.format(
                        self.num_call,
                        round(float(loss), 4),
                        round(float(perf), 4),
                        round(float(self.best_train_perf), 4)), file=log_f)

            if parallel:
                return all_losses
            else:
                return loss


tokenizer = RobertaTokenizer.from_pretrained(model_name)

# @cache_results(cache_fn, _refresh=False)
@cache_results(_cache_fp=None, _refresh=False)
def get_data(data, tokenizer, path=None):
    
    data_bundle = sent140Loader(tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens).my_load(data)
    
    return data_bundle


def construct_few_shot_data(train_data):
    new_train_data = DataSet()
    for ind in range(len(train_data)):
        new_train_data.append(train_data[ind])
        
    new_train_data.set_input("input_ids", "attention_mask", "mask_pos")

    new_train_data.set_target("labels")

    return new_train_data


def construct_true_few_shot_data(train_data, k_shot, n_clients):
    train_label_counts = [{} for _ in range(n_clients)]
    new_train_data = [DataSet() for _ in range(n_clients)]
    all_indices = [_ for _ in range(len(train_data))]
    np.random.shuffle(all_indices)

    for index in all_indices:
        label = train_data[index]['labels']
        if label < 0:
            continue

        if label not in train_label_counts[0]:
            for i in range(n_clients):
                train_label_counts[i][label] = 0


        if train_label_counts[-1][label] < k_shot:
            i = 0
            while train_label_counts[i][label] == k_shot:
                i += 1
            new_train_data[i].append(train_data[index])
            train_label_counts[i][label] += 1


    for i in range(n_clients):
        new_train_data[i].set_input("input_ids", "attention_mask", "mask_pos")
        new_train_data[i].set_target("labels")


#     new_train_data.set_target("labels")
#     new_dev_data.set_target("labels")
    return new_train_data




''' data_raw below should be a dictionary that has the following structure:

{'train': {'usr1': {'sentence': [sentence1, sentence2, ...], 'label': [label1, label2...]}, 
           'usr2': {'sentence': [sentence1, sentence2, ...], 'label': [label1, label2...]},
           'usr3': ...
           ......},
'test': {'usr1': {'sentence': [sentence1, sentence2, ...], 'label': [label1, label2...]}, 
         'usr2': {'sentence': [sentence1, sentence2, ...], 'label': [label1, label2...]},
         'usr3': ...
           ......}
}
The labels are either 0 (negative) or 1 (positive). 
'''
ds_raw = np.load('/path/to/data', allow_pickle=True).item()

counts_train = {0: 0, 1: 0}
ds_train = {}
c_train = 0
for usr in ds_raw['train']:
    print(usr)
#     c_train += 1
#     if c_train > 1000:
#         break
    assert len(ds_raw['train'][usr]['label']) == len(ds_raw['train'][usr]['sentence'])
    if len(ds_raw['train'][usr]['label']) <= 40:
        continue
    for l in ds_raw['train'][usr]['label']:
        counts_train[l] += 1
    
    data_bundle = get_data(ds_raw['train'][usr], tokenizer)
    data = data_bundle.get_dataset('train')
    data = construct_few_shot_data(data)
    
    data.set_pad_val('input_ids', tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    data.set_pad_val('attention_mask', 0)
    
    data_o = data
    data = {
        'input_ids': torch.tensor(data_o['input_ids'].get(list(range(len(data_o))))),
        'attention_mask': torch.tensor(data_o['attention_mask'].get(list(range(len(data_o))))),
        'mask_pos': torch.tensor(data_o['mask_pos'].get(list(range(len(data_o))))),
        'labels': torch.tensor(data_o['labels'].get(list(range(len(data_o))))),
        }
    ds_train[usr] = data
    
counts_test = {0: 0, 1: 0}
ds_test = {}
c_test = 0
for usr in ds_raw['test']:
    print(usr)
#     c_test += 1
#     if c_test > 10:
#         break
    assert len(ds_raw['test'][usr]['label']) == len(ds_raw['test'][usr]['sentence'])
    for l in ds_raw['test'][usr]['label']:
        counts_test[l] += 1
    
    data_bundle = get_data(ds_raw['test'][usr], tokenizer)
    data = data_bundle.get_dataset('train')
    data = construct_few_shot_data(data)
    
    data.set_pad_val('input_ids', tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    data.set_pad_val('attention_mask', 0)
    
#     data_o = data
#     data = {
#         'input_ids': torch.tensor(data_o['input_ids'].get(list(range(len(data_o))))),
#         'attention_mask': torch.tensor(data_o['attention_mask'].get(list(range(len(data_o))))),
#         'mask_pos': torch.tensor(data_o['mask_pos'].get(list(range(len(data_o))))),
#         'labels': torch.tensor(data_o['labels'].get(list(range(len(data_o))))),
#         }
    ds_test[usr] = data    
    
ds = {'train': ds_train, 'test': ds_test}
# torch.save(ds, 'sent140.pt')


usrs_train, usrs_test = list(ds_train.keys()), list(ds_test.keys())




n_clients_all = len(ds['train'].keys())

def split_labels(dataset, ratio=0.5):
    labels = dataset['labels'].cpu().numpy()
    values, counts = np.unique(labels, return_counts=True)
    d = {v:int(c*ratio) for v,c in zip(values, counts)}
    idx_sel, idx_unsel = [], []
    for i, l in enumerate(labels):
        if d[l] > 0:
            idx_sel.append(i)
            d[l] -= 1
        else:
            idx_unsel.append(i)
    dataset_sel, dataset_unsel = {}, {}
    for k in dataset:
        dataset_sel[k] = dataset[k][idx_sel]
        dataset_unsel[k] = dataset[k][idx_unsel]
    return dataset_sel, dataset_unsel

def rand_sel(dataset, n=32):
    labels = dataset['labels'].cpu().numpy()
    idx_sel = np.random.choice(len(labels), min(n,len(labels)), replace=False)
    dataset_sel = {}
    for k in dataset:
        dataset_sel[k] = dataset[k][idx_sel]
    return dataset_sel
        

model_forward_api = LMForwardAPI(
    model_name=model_name,
    n_prompt_tokens=n_prompt_tokens,
    task_name='None',
    # save_path=save_path,
    loss_type=loss_type,
    init_prompt_path=init_prompt_path
)

cma_opts = {
    'seed': seed,
    'popsize': popsize,
    'maxiter': 10,
    'verbose': -1,
}
if bound > 0:
    cma_opts['bounds'] = [-1 * bound, 1 * bound]
# es = cma.CMAEvolutionStrategy(intrinsic_dim * [0], sigma, inopts=cma_opts)
# print('Population Size: {}'.format(es.popsize))
print('{} Evaluation.'.format('Parallel' if parallel else 'Serial'))

# opt = cma.CMAOptions()
model_forward_api.model.descrete = None
model_forward_api.model.m = False

embedding = model_forward_api.model.roberta.embeddings.word_embeddings.weight.detach().clone().cpu()
embedding_m = embedding - torch.mean(embedding, dim=0, keepdim=True)
mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
mu = 0.0
std = std_hat / (np.sqrt(intrinsic_dim) * args.sigma)
m = embedding.mean(0).view(1,-1).numpy()
            

if model_forward_api.model.descrete is None:
    model_forward_api.init_prompt = torch.zeros(n_prompt_tokens * model_forward_api.config.hidden_size)
else:
    offset = 1000
    indices_offset = torch.tensor(list(range(offset, offset + n_prompt_tokens)))
    if model_forward_api.model.m:
        prompt_init = embedding_m[indices_offset].numpy()
    else:
        prompt_init = embedding[indices_offset].numpy()
    model_forward_api.init_prompt = torch.tensor( prompt_init ).view(-1)
    


offset = 1000
indices_offset = torch.tensor(list(range(offset, offset + n_prompt_tokens)))
prompt_offset = embedding[indices_offset]

prompts_all = {}
for usr in ds['train']:
    prompts_all[usr] = copy.deepcopy(prompt_offset)
prompts_avg_old = copy.deepcopy(prompt_offset)

# embedding_avg = embedding.mean(0)
# embedding_logits = torch.matmul( embedding - embedding_avg.view(1,-1), torch.tensor(pca_comp).transpose(0,1) )
# embedding_perm = embedding[torch.tensor(np.random.permutation(len(embedding)))]
# embedding_perm_logits = torch.matmul( embedding - embedding_perm, torch.tensor(pca_comp).transpose(0,1) )
# breakpoint()
    
    
t = None
K = 5
save_path = f"checkpoints/"
os.makedirs(save_path, exist_ok=True)

random.shuffle(usrs_train)
n_clients_r = 50
c = 0
n_rounds = 100

for r in range(n_rounds):
    
    if (c + n_clients_r) > len(usrs_train):
        usrs_train = list(ds['train'].keys())
        random.shuffle(usrs_train)
        c = 0
    usrs_train_r = usrs_train[c:c+n_clients_r]
    c += n_clients_r
    
    for usr in usrs_train_r:
        #         dataset_sel, dataset_unsel = split_labels(train_data_all[i])
#         dataset_unsel = train_data_all[i]
        dataset_unsel = rand_sel(ds['train'][usr], n=32)
#         breakpoint()
        for step_r in range(steps_per_round):
            prompt_i = prompts_all[usr]
            t = np.random.choice(n_prompt_tokens)
            model_forward_api.init_prompt = prompt_i - prompt_offset
           
            prompt_i.requires_grad = False
#             loss_sel = model_forward_api.eval_fed(train_data=dataset_sel, test_data=None, task_name=task_name)
#             loss_unsel = model_forward_api.eval_fed(train_data=dataset_unsel, test_data=None, task_name=task_name)
            loss_unsel = model_forward_api.eval_fed(train_data=dataset_unsel, test_data=None, task_name='sent140')
#             indices_d = np.random.choice(1024, grad_trials)
#             pcas = pca_comp[indices_d ,:]
#             grads_est = []
#             for ind_d in indices_d:
#                 model_forward_api.init_prompt = prompt_i - prompt_offset
#                 res = epsilon * torch.tensor(pca_comp[ind_d,:])
#                 model_forward_api.init_prompt[t] += res
# #             model_forward_api.init_prompt[t,ind_d] += epsilon
#                 loss_res = model_forward_api.eval_fed(train_data=dataset_sel, test_data=None, task_name=task_name)
#                 grad_est = abs((loss_sel - loss_res) / epsilon)
#                 grads_est.append(grad_est)
        
        
            losses = [loss_unsel]
#             grads = [np.mean(grads_est).item()]
            p = torch.tensor(prompt_i[t])
            logits_t = torch.matmul(embedding, p.view(-1,1)).view(-1)
            probs_t = torch.nn.functional.softmax(logits_t, dim=-1)
            indices_t = torch.topk(probs_t, K)[1]
            for ind_r in indices_t:
                p_r = embedding[ind_r]
#             prompt_i_r = copy.deepcopy(prompt_i)
                prompt_i_r = copy.deepcopy(prompt_i.cpu().detach().numpy())
                prompt_i_r[t] = p_r.cpu().detach().numpy()
                prompt_i_r = torch.tensor( prompt_i_r, requires_grad=False )
                model_forward_api.init_prompt = prompt_i_r - prompt_offset
#                 loss_sel = model_forward_api.eval_fed(train_data=dataset_sel, test_data=None, task_name=task_name)
#                 loss_unsel = model_forward_api.eval_fed(train_data=dataset_unsel, test_data=None, task_name=task_name)
                loss_unsel = model_forward_api.eval_fed(train_data=dataset_unsel, test_data=None, task_name='sent140')
                losses.append(loss_unsel)
#                 grads_est = []
#                 for ind_d in indices_d:
#                     model_forward_api.init_prompt = prompt_i_r - prompt_offset
#                     res = epsilon * torch.tensor(pca_comp[ind_d,:])
#                     model_forward_api.init_prompt[t] += res
#                     loss_res = model_forward_api.eval_fed(train_data=dataset_sel, test_data=None, task_name=task_name)
#                     grad_est = abs((loss_sel - loss_res) / epsilon)
#                     grads_est.append(grad_est)
#                 grads.append(np.mean(grads_est).item())
            
            
            losses = np.array(losses)
#             grads = np.array(grads)
        
#             if r < 20:
#                 m_losses = losses - step_size * grads
#             else:
#                 m_losses = losses
            m_losses = losses #- step_size * grads
#         breakpoint()
            i_min = np.argmin(m_losses).item()
    
            print(t)
            print(m_losses)
#             if m_losses[0] != m_losses[1]:
#                 breakpoint()
            print(f'Round: {r}, m_loss: {m_losses[i_min]}, loss {losses[i_min]}')
            with open(os.path.join(save_path, 'log.txt'), 'a') as f:
                print(t, file=f)
                print(m_losses, file=f)
                print(f'Round: {r}, m_loss: {m_losses[i_min]}, loss {losses[i_min]}', file=f)
        
            if i_min > 0:
                ind_r = indices_t[i_min-1]
                p_r = embedding[ind_r]
                prompt_i_r = copy.deepcopy(prompt_i.cpu().detach().numpy())
                prompt_i_r[t] = p_r.cpu().detach().numpy()
                prompt_i = torch.tensor( prompt_i_r, requires_grad=False )
                prompts_all[usr] = prompt_i
#         breakpoint()
    prompts_avg = torch.stack([prompts_all[usr] for usr in usrs_train_r], dim=0).mean(0)
    prompts_avg_diff = prompts_avg - prompts_avg_old

    embedding_ext = torch.cat([embedding, prompts_avg_old], dim=0)
    prompts_avg_diff_recons = []
    res = []
    for p_id in range(n_prompt_tokens):
        emb_t = prompts_avg_diff[p_id].view(1,-1).numpy()
        indices = list(range(embedding_ext.size(0)))
        d = 0.2 * (1 / (2 * 1024))
#         d = 0 * (1 / (2 * 1024))
        for D in [100, 5, 5]:
#         for D in [1000, 100, 50, 5]:
            emb_n = embedding_ext.numpy()[indices,:] / np.linalg.norm(embedding_ext.numpy()[indices,:], axis=-1, keepdims=True)
            A = np.transpose(emb_n)
            b = emb_t.flatten()
            clf = linear_model.Lasso(alpha=d)
            clf.fit(A, b)
            theta = np.array(clf.coef_)
            emb_t_recons = A@theta 
            print( np.linalg.norm(emb_t - emb_t_recons.flatten()) )
            res.append(np.linalg.norm(emb_t - emb_t_recons.flatten()))
#             print( np.sqrt( ( (emb_t - emb_t_recons.flatten())**2 ).mean() ) )
            if D == 5:
                d = 0
            indices = [indices[i] for i in np.argsort(np.abs(theta))[-D:] ]
#             print(indices[-30:])
#         breakpoint()
        prompts_avg_diff_recons.append(torch.tensor(emb_t_recons).view(-1))
    
    prompts_avg_diff_recons = torch.stack(prompts_avg_diff_recons, dim=0)
#     breakpoint()
    prompts_avg_new = prompts_avg_old + prompts_avg_diff_recons
    prompts_avg_old = copy.deepcopy(prompts_avg_new)
    
    
    

    prompts_all = {}
    for usr in ds['train']:
#         prompts_all.append(copy.deepcopy(prompts_all_s_new[i]))
#         prompts_all.append(copy.deepcopy(prompt_avg))
        prompts_all[usr] = copy.deepcopy(prompts_avg_new)
    if r % 10 == 0:
#         torch.save(prompts_all_s_new, os.path.join(save_path, f'{r}.pt'))
        torch.save(prompts_avg_new, os.path.join(save_path, f'{r}.pt'))
        
            

            
            
            

# for r in range(n_rounds):
#     prompt_embeddings_all = []
#     for n in range(n_clients):
#         print('\n\nRound: {}, Client: {}'.format(r, n))
#         if log_file is not None:
#             with open(log_file, 'a') as log_f:
#                 print('\n\nRound: {}, Client: {}'.format(r, n), file=log_f)
#         model_forward_api.best_train_perf = 0.0
#         model_forward_api.num_call = 0
        
            
#         es = cma.CMAEvolutionStrategy(intrinsic_dim * [0], sigma, inopts=cma_opts)
#         while not es.stop():
#             solutions = es.ask()
#             fitnesses = [model_forward_api.eval_fed(intrincs=x, train_data=train_data[n], test_data=None, log_file=log_file) for x in solutions]
#             es.tell(solutions, fitnesses)
#         solutions = np.stack(solutions, axis=0)
#         solutions_mean = torch.tensor( solutions.mean(0) ).type(torch.float32)
#         with torch.no_grad():
#             prompt_embeddings_all.append( model_forward_api.linear(solutions_mean) )
#     prompt_embeddings_all = torch.stack(prompt_embeddings_all, dim=0)
#     prompt_embeddings_mean = prompt_embeddings_all.mean(0)
#     model_forward_api.init_prompt = model_forward_api.init_prompt + prompt_embeddings_mean
#     print('Evaluate on test data...')
#     test_acc = model_forward_api.eval_fed(test_data=test_data, log_file=log_file)
#     print('Test acc: {}'.format(round(test_acc, 4)))
#     if log_file is not None:
#         with open(log_file, 'a') as log_f:
#             print('Evaluate on test data...', file=log_f)
#             print('Test acc: {}'.format(round(test_acc, 4)), file=log_f)
        
        

# start_time = time.time()
# while not es.stop():
# #     breakpoint()
#     solutions = es.ask()
#     if parallel:
#         fitnesses = model_forward_api.eval(solutions)
#     else:
#         fitnesses = [model_forward_api.eval(x) for x in solutions]
#     es.tell(solutions, fitnesses)
#     # es.logger.add()  # write data to disc to be plotted
#     # es.disp()
# end_time = time.time()
# print('Done. Elapsed time: {} (mins)'.format((end_time - start_time) / 60))
# print('Evaluate on test data...')
# test_acc = model_forward_api.eval(test_data=test_data)
# print('Test acc: {}'.format(round(test_acc, 4)))
# fitlog.finish()
