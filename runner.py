# import main
from make_dataloaders import SICKDataset, collate_fn
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoConfig
from torch.utils.data import DataLoader
from utils import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

import torch
from tqdm import tqdm, trange
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
END = 9

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    return tokenizer, model

def predict(tokenizer, model, dataloader):
    predict_label = []
    golden_label = []
    tokens = []

    for sample in tqdm(dataloader):
        tokenized_sentence = sample['sentence']
        tokenized_sentence = tokenized_sentence.to(DEVICE)
        with torch.no_grad():
            predictions = model(tokenized_sentence)[0]
        # print(predictions.shape)
        mask_predictions = torch.softmax(predictions.squeeze()[sample['mask_index']], dim=-1)

        # _, ids = torch.topk(mask_predictions, k=5)
        # token = tokenizer.convert_ids_to_tokens(ids)
        # tokens.append(token)
        # print(_)
        # print(ids)
        # print(token)
        # print("golden: {}".format(sample['label']))

        predict_label.append(torch.argmax(mask_predictions[torch.tensor(sample['target_ids'])]).item())
        # print(predict_prob[-1])
        golden_label.append(sample['label'])

    acc = bi_accuracy(predict_label, golden_label)
    return predict_label, golden_label, acc

def vote(all_predicts, accs):
    results = []
    all_predicts = torch.transpose(torch.tensor(all_predicts), 0, 1).tolist()
    alpha = accs
    thres = 0.5
    for predicts in all_predicts:
        score_0 = 0
        score_2 = 0
        for pred, (a0, a2) in zip(predicts, alpha):
            if pred == 0:
                score_0 += a0
            else:
                score_2 += a2
        if score_0 / sum(alpha[0]) > thres and score_2 / sum(alpha[1]) > thres:
            results.append(1)
        elif score_0 / sum(alpha[0]) > thres:
            results.append(0)
        elif score_2 / sum(alpha[1]) > thres:
            results.append(2)
        else:
            results.append(1)
    return results

def bi_accuracy(predict, golden_label):
    correct = 0
    wrong = 0
    correct_0 = 0
    wrong_0 = 0
    correct_2 = 0
    wrong_2 = 0
    for pred, label in zip(predict, golden_label):
        if label == 1:
            continue
        if pred == 1:
            pred = 2
        if pred == label:
            correct += 1
            if label == 0:
                correct_0 += 1
            else:
                correct_2 += 1
        else:
            wrong += 1
            if label == 0:
                wrong_0 += 1
            else:
                wrong_2 += 1

    acc = correct / (correct + wrong + 1e-4)
    acc_0 = correct_0 / (correct_0 + wrong_0 + 1e-4)
    acc_2 = correct_2 / (correct_2 + wrong_2 + 1e-4)
    print("The accuracy score is: {:.4f}".format(acc))
    print("Label 0 is: {:.4f}".format(acc_0))
    print("Label 2: {:.4f}".format(acc_2))
    return acc_0, acc_2

def accuracy(predict, golden_label):
    correct = 0
    wrong = 0
    correct_0 = 0
    wrong_0 = 0
    correct_1 = 0
    wrong_1 = 0
    correct_2 = 0
    wrong_2 = 0
    for pred, label in zip(predict, golden_label):
        if pred == label:
            correct += 1
            if label == 0:
                correct_0 += 1
            elif label == 1:
                correct_1 += 1
            else:
                correct_2 += 1
        else:
            wrong += 1
            if label == 0:
                wrong_0 += 1
            elif label == 1:
                wrong_1 += 1
            else:
                wrong_2 += 1

    print("The accuracy score is: {:.4f}".format(correct / (correct + wrong + 1e-4)))
    print("Label 0 is: {:.4f}".format(correct_0 / (correct_0 + wrong_0 + 1e-4)))
    print("Label 1: {:.4f}".format(correct_1 / (correct_1 + wrong_1 + 1e-4)))
    print("Label 2: {:.4f}".format(correct_2 / (correct_2 + wrong_2 + 1e-4)))

def valid(dataset_name, tokenizer, model):
    all_predicts = []
    golden = []
    accs = []
    for i in trange(1, END):
        # for j in range(i, i+4):
        template_path = 'templates/{}/template{}/template_valid.json'.format(dataset_name, i)
        dataset = SICKDataset(tokenizer, template_path)
        print("Dataset Path: {}".format(template_path))
        print("Dataset Example: ")
        print(dataset[0]['sentence_str'])
        # assert False
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        pred, gold, acc = predict(tokenizer, model, dataloader)
        if min(acc) < 0.20:
            print("Useless prediction, skip!")
        all_predicts.append(pred)
        golden = gold
        accs.append(acc)
    #     accuracy(vote(all_predicts, accs), golden)
    return all_predicts, golden, accs

def test(dataset_name, tokenizer, model):
    all_predicts = []
    golden = []
    accs = []
    for i in trange(1, END):
        # for j in range(i, i+4):
        template_path = 'templates/{}/template{}/template_test.json'.format(dataset_name, i)
        dataset = SICKDataset(tokenizer, template_path)
        print("Dataset Path: {}".format(template_path))
        print("Dataset Example: ")
        print(dataset[0]['sentence_str'])
        # assert False
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        pred, gold, acc = predict(tokenizer, model, dataloader)
        all_predicts.append(pred)
        golden = gold
        accs.append(acc)
    #     accuracy(vote(all_predicts, accs), golden)
    return all_predicts, golden

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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model = model.to(DEVICE)

    val_predicts, val_golden, val_accs = valid(dataset_name, tokenizer, model)

    val_predicts = np.array(val_predicts).T
    print(val_accs)
    val_accs = np.array(val_accs).reshape(1, -1)
    print(val_accs.shape)
    _val_accs = np.repeat(val_accs, val_predicts.shape[0], axis=0)
    val_predicts = np.concatenate((val_predicts, _val_accs), axis=1)
    val_golden = np.array(val_golden)
    print(val_predicts.shape)
    print(val_golden.shape)

    DT = DecisionTreeClassifier()
    DT.fit(val_predicts, val_golden)

    # Make predictions on the test set
    test_predicts, test_golden = test(dataset_name, tokenizer, model)
    _test_predicts = np.array(test_predicts).T
    _test_predicts = np.concatenate((_test_predicts, np.repeat(val_accs, _test_predicts.shape[0], axis=0)), axis=1)
    _test_golden = np.array(test_golden)
    print(_test_predicts.shape)
    print(_test_golden.shape)

    predictions = DT.predict(_test_predicts)
    print(predictions)
    accuracy(predictions, test_golden)