# import main
from make_dataloaders import SICK_t5_Dataset, collate_fn
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from utils import *

import torch
from tqdm import tqdm
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reformat_prediction(prediction, target):
    for predict in prediction:
        if predict.lower().startswith(target[0]):
            return 0
        elif predict.lower().startswith(target[1]):
            return 1
        else:
            return 2
def predict(tokenizer, model, dataloader):
    predict_label = []
    golden_label = []

    for sample in tqdm(dataloader):
        tokenized_sentence = sample['sentence']
        tokenized_sentence = tokenized_sentence.to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(input_ids=tokenized_sentence,
                                     num_beams=200, num_return_sequences=10,
                                     max_length=10)
        predict_token = [tokenizer.decode(output[2:]) for output in outputs]
        print(predict_token)
        predict = reformat_prediction(predict_token, sample['target_label'])
        print("-")
        predict_label.append(predict)
        golden_label.append(sample['label'])

    accuracy(predict_label, golden_label)

def accuracy(predict, golden_label):
    correct = 0
    wrong = 0

    for pred, label in zip(predict, golden_label):
        if pred == int(label):
            correct += 1
        else:
            wrong += 1

    print("The accuracy score is: {:.4f}".format(correct / (correct + wrong)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", default=None, type=str, required=True, help="Input dataset name."
    )
    parser.add_argument(
        "--dataset", default=None, type=str, required=True, help="Input dataset name."
    )
    parser.add_argument(
        "--template", default=None, type=str, required=True, help="Input dataset name."
    )
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset
    template_num = args.template

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    config = T5Config.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(DEVICE)


    template_path = 'templates/{}/template{}/template_valid.json'.format(dataset_name, template_num)
    dataset = SICK_t5_Dataset(tokenizer, template_path)
    print("Dataset Example: ")
    print(dataset[0])
    # assert False
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    predict(tokenizer, model, dataloader)
