from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import argparse
from make_dataloaders import SICK_pipeline_Dataset


def predict(model_name, dataloader):
    predict_label = []
    golden_label = []

    target_tokens = ["right", "wrong", "unknown"]
    label_to_id = {target_tokens[0].lower(): 0, target_tokens[1].lower(): 2, target_tokens[2].lower(): 1}

    fill_masker = pipeline("fill-mask", model=model_name, targets=target_tokens)
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
    accuracy(predict_label, golden_label)


def accuracy(predict, golden_label):
    correct = 0
    wrong = 0

    for pred, label in zip(predict, golden_label):
        if pred == int(label):
            correct += 1
        else:
            wrong += 1

    print("Total: {}. Correct: {}. Wrong: {}".format(correct + wrong, correct, wrong))
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelWithLMHead.from_pretrained(model_name)

    template_path = 'templates/{}/template{}/template_valid.json'.format(dataset_name, template_num)
    dataset = SICK_pipeline_Dataset(tokenizer, template_path)
    print("Dataset Example: ")
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    predict(model_name, dataloader)
