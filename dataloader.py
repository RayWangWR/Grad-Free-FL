import datasets
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle
from functools import partial
from transformers import RobertaTokenizer
from datasets import Dataset
import pandas as pd
import os


def convert_to_features(example_batch, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'])
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], add_special_tokens=False)
    mask_pos = []
    for input_ids in input_encodings['input_ids']:
        mask_pos.append(input_ids.index(tokenizer.mask_token_id))
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'mask_pos': mask_pos,
        'labels': target_encodings['input_ids'],
    }

    return encodings



class sent140Loader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "bad",
            1: "great",
        }
#         self.path = path

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s . It was %s .' % (prompt, example['sentence'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s . It was %s .' % (example['sentence'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, d) -> DataSet:
        # load dataset with Huggingface's Datasets
#         dataset = datasets.load_dataset('glue', 'sst2', split=split)

        dataset = Dataset.from_dict(d)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
#         print('Example in {} set:'.format(split))
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, data) -> DataBundle:
        datasets = {'train': self._load(data)}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle
    
    
