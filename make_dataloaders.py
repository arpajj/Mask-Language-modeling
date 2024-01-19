import json
import torch
from torch.utils.data import DataLoader, Dataset
from utils import *
from transformers import AutoTokenizer

def load_json(path):
    with open(path, 'r') as json_file:
        dataset = json.load(json_file)
    return dataset

class SICKDataset(Dataset):
    def __init__(self, tokenizer, path):
        self.tokenizer = tokenizer
        self.data = self.process(load_json(path))

    def process(self, data):
        new_data = []
        for sample in data[:1000]:
            sample['sentence'] = sample['sentence'].replace("[MASK]",
                                                            self.tokenizer.convert_ids_to_tokens(
                                                                self.tokenizer.mask_token_id))
            tokenized_sentence = self.tokenizer.encode(sample['sentence'], return_tensors='pt')
            mask_index = torch.where(tokenized_sentence == self.tokenizer.mask_token_id)[-1].squeeze().tolist()
            target_ids = [self.tokenizer.encode(target)[1] for target in sample['predict_label']]
            label = sample['label']
            new_data.append({
                "sentence": tokenized_sentence,
                "sentence_str": sample['sentence'],
                "mask_index": mask_index,
                "target_label": sample['predict_label'],
                "target_ids": target_ids,
                "label": label
            })
        return new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SICK_pipeline_Dataset(Dataset):
    def __init__(self, tokenizer, path):
        self.tokenizer = tokenizer
        self.data = self.process(load_json(path))

    def process(self, data):
        new_data = []
        for sample in data[:1000]:
            sample['sentence'] = sample['sentence'].replace("[MASK]",
                                                    self.tokenizer.convert_ids_to_tokens(self.tokenizer.mask_token_id))
            # tokenized_sentence = self.tokenizer.encode(sample['sentence'], return_tensors='pt')
            # mask_index = torch.where(tokenized_sentence == self.tokenizer.mask_token_id)[-1].squeeze().tolist()
            label = sample['label']
            new_data.append({
                # "sentence": tokenized_sentence,
                "sentence": sample['sentence'],
                # "mask_index": mask_index,
                "label": str(label)
            })
        return new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # only for batchsize == 1
    return {
        "sentence": batch[0]['sentence'],
        "sentence_str": batch[0]['sentence_str'],
        "mask_index": batch[0]['mask_index'],
        "target_label": batch[0]['target_label'],
        "target_ids": batch[0]['target_ids'],
        "label": batch[0]['label']
    }

class SICK_t5_Dataset(Dataset):
    def __init__(self, tokenizer, path):
        self.tokenizer = tokenizer
        self.data = self.process(load_json(path))

    def process(self, data):
        new_data = []
        for sample in data:
            sample['sentence'] = sample['sentence'].replace("[MASK]", "<extra_id_0>")
            tokenized_sentence = self.tokenizer.encode(sample['sentence'], add_special_tokens=True, return_tensors='pt')
            mask_index = torch.where(tokenized_sentence ==
                                     self.tokenizer.convert_tokens_to_ids("<extra_id_0>"))[-1].squeeze().tolist()
            target_ids = [self.tokenizer.encode(target) for target in sample['predict_label']]
            label = sample['label']
            new_data.append({
                "sentence": tokenized_sentence,
                "sentence_str": sample['sentence'],
                "mask_index": mask_index,
                "target_label": sample['predict_label'],
                "target_ids": target_ids,
                "label": label
            })
        return new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    # print(tokenizer.encode("[MASK]"))
    # print(tokenizer.mask_token_id)
    # print(tokenizer.convert_ids_to_tokens(tokenizer.mask_token_id))
    my_template_path = 'templates/template1/template_valid.json'

    # Create the Dataset object and the DataLoader
    my_custom_dataset = SICKDataset(tokenizer, my_template_path)
    my_dataloader = DataLoader(my_custom_dataset, batch_size=2, shuffle=False)
    for x in my_dataloader:
        print(x)
        assert False