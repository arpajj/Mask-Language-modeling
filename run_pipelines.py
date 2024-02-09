from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import argparse
from make_dataloaders import SICK_pipeline_Dataset
import os

def predict(model_name, dataloader):
    predict_label = []
    golden_label = []

    target_tokens = ["true", "false", "unknown"]
    label_to_id = {target_tokens[0].lower(): 0, target_tokens[1].lower(): 2, target_tokens[2].lower(): 1}

    fill_masker = pipeline("fill-mask", model=model_name, targets=target_tokens, device=0)
    for sample in tqdm(dataloader):
        results = fill_masker(sample['sentence'])
        # print(results)
        # assert False
        target_prob = {}
        for result in results:
            for pair in result:
                target_prob[pair['token_str'].lower()] = pair['score']
            final_prediction = max(target_prob, key=target_prob.get)
            # assert False
            predict_label.append(label_to_id[final_prediction])
        golden_label.extend(sample['label'])
    res = accuracy(predict_label, golden_label)
    return res

def accuracy(predict, golden_label):
    correct = 0
    wrong = 0
    detail_c = [0, 0, 0]
    detail_w = [0, 0, 0]
    for pred, label in zip(predict, golden_label):
        if pred == int(label):
            correct += 1
            detail_c[int(label)] += 1
        else:
            wrong += 1
            detail_w[int(label)] += 1
    t = correct / (correct + wrong)
    t1 = detail_c[0] / (detail_c[0] + detail_w[0])
    t2 = detail_c[1] / (detail_c[1] + detail_w[1])
    t3 = detail_c[2] / (detail_c[2] + detail_w[2])
    # print("Total: {}. Correct: {}. Wrong: {}".format(correct + wrong, correct, wrong))
    print("The accuracy score is: {:.4f}".format(t))
    print("The entailment acc score is: {:.4f}".format(t1))
    print("The neutral acc score is: {:.4f}".format(t2))
    print("The contradiction acc score is: {:.4f}".format(t3))
    return t, t1, t2, t3

if __name__ == '__main__':
    # command = os.popen("nvidia-smi -q -d PIDS | grep Processes")
    # lines = command.read().split("\n")
    # free_gpu = []
    # for i in range(len(lines)):
    #     if "None" in lines[i]:
    #         free_gpu.append(i)
    # print(free_gpu)
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
    # model = AutoModelWithLMHead.from_pretrained(model_name)

    result = []
    for i in range(1, 17):
        template_path = 'templates/{}/template{}/template_valid.json'.format(dataset_name, i)
        dataset = SICK_pipeline_Dataset(tokenizer, template_path)
        print(f"Template: {i}")
        # print("Dataset Example: ")
        # print(dataset[0])
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        res = predict(model_name, dataloader)
    result.append(res)
    print(result)