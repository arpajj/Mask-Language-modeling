# import main
from make_dataloaders import MyDataset, collate_fn
from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from utils import *

import torch
from tqdm import tqdm
import argparse

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    return tokenizer, model

def predict(tokenizer, model, dataloader):
    predict_prob = []
    golden_label = []

    true_idx = tokenizer.convert_tokens_to_ids("true")
    false_idx = tokenizer.convert_tokens_to_ids("false")
    unknown_idx = tokenizer.convert_tokens_to_ids("unknown")
    for sample in tqdm(dataloader):
        tokenized_sentence = sample['sentence']
        with torch.no_grad():
            predictions = model(tokenized_sentence)[0]
        # print(predictions.shape)
        mask_predictions = torch.softmax(predictions.squeeze()[sample['mask_index']], dim=-1)
        # print([sample['mask_index']])
        # print(mask_predictions.shape)
        predict_prob.append(mask_predictions[torch.tensor([true_idx, false_idx, unknown_idx])])
        # print(predict_prob[-1])
        golden_label.append(sample['label'])
    accuracy(predict_prob, golden_label)

def accuracy(predict, golden_label):
    correct = 0
    wrong = 0

    for probs, label in zip(predict, golden_label):
        if torch.argmax(probs).item() == label:
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
        "--template", default=None, type=str, required=True, help="Input dataset name."
    )
    args = parser.parse_args()

    model_name = args.model
    template_num = args.template

    tokenizer, model = load_model(model_name)
    # tokenizer_roberta, model_roberta = load_model("roberta-base")
    # tokenizer_gpt2, model_gpt2 = load_model("gpt2")

    template_path = 'templates/template{}/template_valid.json'.format(template_num)
    my_custom_dataset = MyDataset(tokenizer, template_path)
    my_dataloader = DataLoader(my_custom_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    predict(tokenizer, model, my_dataloader)
    # all_labels = [entry['label'] for entry in dataset_v2]
    #
    # main.plot_distribution(all_labels)

    # models = [model_bert, model_roberta, model_gpt2]
    # tokenizers = [tokenizer_bert, tokenizer_roberta, tokenizer_gpt2]
    # probs_of_models_over_dateset = []
    # for i, (model, tokenizer) in enumerate(zip(models, tokenizers)):
    #     probs_over_dateset = main.prob_distribution_over_vocab_with_batch(model, tokenizer, my_dataloader)
    #     # probs_of_models_over_dateset.append(probs_over_dateset)
    #     print("Finished model {}: {} \n".format(i + 1, model.name_or_path.split("-")[0]))
    #     results = [inner_list for outer_list in probs_over_dateset for inner_list in outer_list]
    #     accuracy(results)
    #
    # results_bert_flattened = [inner_list for outer_list in probs_of_models_over_dateset[0] for inner_list in outer_list]
    # results_roberta_flattened = [inner_list for outer_list in probs_of_models_over_dateset[1] for inner_list in outer_list]
    # results_gpt2_flattened = [inner_list for outer_list in probs_of_models_over_dateset[2] for inner_list in outer_list]
    #
    # accuracy(results_bert_flattened)
    # accuracy(results_roberta_flattened)
    # accuracy(results_gpt2_flattened)
