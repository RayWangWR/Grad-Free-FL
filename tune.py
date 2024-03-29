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
parser.add_argument("--model_name", default='roberta-large',
                    choices=['roberta-base', 'roberta-large',
                             'bert-base-uncased', 'bert-large-uncased',
                             'google/electra-base-generator', 'google/electra-large-generator',
                             'facebook/bart-base', 'facebook/bart-large',
                             't5-small', 't5-base', 't5-large', 't5-3b',
                             'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                             'fnlp/cpt-large'], type=str)
parser.add_argument("--task_name", default='sst2', type=str)
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

# log_dir = './v2_logs'
# fitlog.set_log_dir(log_dir)
# fitlog.commit(__file__, fit_msg=save_path)
# fitlog.add_hyper(args)
# fitlog.add_hyper_in_file(__file__)

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
    
            
    def eval(self, prompt_embedding=None, train_data=None, test_data=None, dev_data=None, task_name=None, return_devloss=False):
        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
#         if test_data is None:
#             bsz = len(dev_data['input_ids'])  # batch size of dev data is the orignal batch size of training data
#         else:
#             bsz = batch_size  # for test data
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
            self.linear.to(self.model.device)
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32).to(self.model.device)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(n_prompt_tokens, -1)#.repeat(bsz, 1, 1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        self.model.set_prompt_embedding(prompt_embedding)

        if isinstance(test_data, DataSet):
#             if prompt_embedding.shape[0] > bsz:
#                 raise ValueError('Provide a single prompt embedding for testing.')
            test_tester = Tester(data=test_data, model=self.model, metrics=self.metric[task_name], batch_size=batch_size, num_workers=1, device=device, use_tqdm=True)
            results = test_tester.test()
            test_acc = results[self.metric_name[task_name]][self.metric_key[task_name]]
            # fitlog.add_best_metric(test_acc, name='test_acc')
            return test_acc
        else:
            for k, v in train_data.items():
                train_data[k] = v.to(device)
            with torch.no_grad():
                bs = 2
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
                    tmp_loss, tmp_perf = self.calc_metric(tmp_logits, tmp_target)
                    all_losses.append(tmp_loss)
                    all_perfs.append(tmp_perf)
                loss = min(all_losses)
                best_sol = all_losses.index(loss)  # argmin
                perf = all_perfs[best_sol]  # corresponding performance
                tmp_prompt = tmp_prompt[best_sol]  # numpy.ndarray
                prompt_embedding = pe_list[best_sol]  # to be prepended to the input
            else:  # single query
                loss, perf = self.calc_metric(logits, train_data['labels'], task_name=task_name)
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

            if (self.num_call == 1 or self.num_call % self.eval_every == 0) and  return_devloss:
                print('********* Evaluated on dev set *********')
                if parallel:  # if we have multiple queries, use the one that achieves minimal loss
                    self.model.set_prompt_embedding(prompt_embedding)
                for k, v in dev_data.items():
                    dev_data[k] = v.to(device)
                with torch.no_grad():
                    bs = 2
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


                dev_loss, dev_perf = self.calc_metric(logits, train_data['labels'], task_name=task_name)
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
                if return_devloss:
                    return loss, dev_loss
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
                    bs = 1
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


def split_labels(data, n=32):
    labels = data['label']
    values, counts = np.unique(labels, return_counts=True)
    d = {}
    for v,c in zip(values, counts):
        n_sel = int(n * c * 1.0 / counts.sum())
        if n_sel == 0:
            n_sel = 1
        d[v] = n_sel
#     d = {v:int(c*ratio) for v,c in zip(values, counts)}
    indices = np.random.permutation(len(labels))
    data_sel, data_unsel = {'sentence': [], 'label': []}, {'sentence': [], 'label': []}
    for idx in indices:
        l = labels[idx]
        if d[l] > 0:
            data_sel['sentence'].append(data['sentence'][idx])
            data_sel['label'].append(data['label'][idx])
            d[l] -= 1
        else:
            data_unsel['sentence'].append(data['sentence'][idx])
            data_unsel['label'].append(data['label'][idx])
    return data_sel, data_unsel


''' data_raw below (same as in joint_training.py) should be a dictionary that has the following structure:

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
usrs_train, usrs_test = list(usrs['train'].keys()), list(usrs['test'].keys())

ds_train, ds_test = {}, {}
for usr in usrs_test:
    print(usr)
#     c_test += 1
#     if c_test > 10:
#         break
    assert len(ds_raw['train'][usr]['label']) == len(ds_raw['train'][usr]['sentence'])
#     for l in ds_raw['test'][usr]['label']:
#         counts_test[l] += 1
    
    data_train_raw, data_test_raw = split_labels(ds_raw['train'][usr])
    data_bundle_train = get_data(data_train_raw, tokenizer)
    data_bundle_test = get_data(data_test_raw, tokenizer)
    
    data_train = data_bundle_train.get_dataset('train')
    data_test = data_bundle_test.get_dataset('train')
    data_train, data_test = construct_few_shot_data(data_train), construct_few_shot_data(data_test)
    
    data_train.set_pad_val('input_ids', tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    data_train.set_pad_val('attention_mask', 0)
    data_test.set_pad_val('input_ids', tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    data_test.set_pad_val('attention_mask', 0)
    
    data_train_o = data_train
    data_train = {
        'input_ids': torch.tensor(data_train_o['input_ids'].get(list(range(len(data_train_o))))),
        'attention_mask': torch.tensor(data_train_o['attention_mask'].get(list(range(len(data_train_o))))),
        'mask_pos': torch.tensor(data_train_o['mask_pos'].get(list(range(len(data_train_o))))),
        'labels': torch.tensor(data_train_o['labels'].get(list(range(len(data_train_o))))),
        }
    
    ds_train[usr] = data_train
    ds_test[usr] = data_test   

ds = {'train': ds_train, 'test': ds_test}
    
        
# n_clients_all = len(train_data_all)

model_forward_api = LMForwardAPI(
    model_name=model_name,
    n_prompt_tokens=n_prompt_tokens,
    task_name='sent140',
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

init_path = 'checkpoints/'
load_round = 100
prompt_init = torch.load(os.path.join(init_path, f'{load_round}.pt'))
log_file = f'{load_round}_post_tuning'
prompts_all = {}
for usr in ds['train']:
#     prompts_all[usr] = copy.deepcopy(prompt_offset)
    prompts_all[usr] = copy.deepcopy(prompt_init)
# prompts_avg_old = copy.deepcopy(prompt_offset)


n_rounds = 1 # do not change this value
for r in range(n_rounds):
    for usr in ds['train']:
        
        model_forward_api.best_train_perf = 0.0
        model_forward_api.best_dev_perf = 0.0
        model_forward_api.best_prompt = None
#         breakpoint()

        print('\n')
        print(usr)
        with open(os.path.join(init_path, log_file), 'a') as f:
            print(usr, file=f)
        prompt_i = prompts_all[usr]
        model_forward_api.init_prompt = (prompt_i - prompt_offset).view(-1).to(model_forward_api.model.device)
        
        cma_opts = {
            'seed': seed,
            'popsize': popsize,
            'maxiter': budget if parallel else budget // popsize,
            'verbose': -1,
            'AdaptSigma': False,
        }
        if bound > 0:
            cma_opts['bounds'] = [-1 * bound, 1 * bound]
        es = cma.CMAEvolutionStrategy(intrinsic_dim * [0], 0.1, inopts=cma_opts)
        model_forward_api.model.descrete = None
        
        s = 0
        while not es.stop():
            s += 1
#             if s > 400:
#                 es.sigma = 0.3
#     breakpoint()
            solutions = es.ask()
            if parallel:
                fitnesses = model_forward_api.eval(solutions)
            else:
                fitnesses = [model_forward_api.eval(x, train_data=ds['train'][usr], dev_data=None, task_name='sent140', return_devloss=False) for x in solutions]
            es.tell(solutions, fitnesses)
    
            if s % 10 == 0 or s in [1, 5, 10, 15, 20, 25]: #or s in [5, 10, 15, 20, 25, 30]:
                solutions_all = np.stack(solutions, axis=0)
                solutions_all_mean = solutions_all.mean(0)
                if s == 1:
                    test_acc_mean = model_forward_api.eval(prompt_embedding=np.array(intrinsic_dim * [0]), test_data=ds['test'][usr], task_name='sent140')
                else:
                    test_acc_mean = model_forward_api.eval(prompt_embedding=solutions_all_mean, test_data=ds['test'][usr], task_name='sent140')

                with open(os.path.join(init_path, log_file), 'a') as f:
                    print(fitnesses, file=f)
                    print('Mean Test acc: {}'.format(round(test_acc_mean, 8)), file=f)
#                     print('Test acc: {}'.format(round(test_acc, 4)), file=f)
        solutions_all = np.stack(solutions, axis=0)
        solutions_all_mean = solutions_all.mean(0)
        prompt_embedding = torch.tensor(solutions_all_mean).type(torch.float32).to(device)  # z
        prompt_embedding = model_forward_api.linear(prompt_embedding).view(prompt_i.shape).cpu().detach()
        prompts_all[usr] = copy.deepcopy(prompt_embedding + prompt_i)
