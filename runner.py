from Helpers import main
from make_dataloaders import MyDataset
from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from utils import *


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    return tokenizer, model


# # Load the BERT tokenizer and model for masked language modeling
# tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
# model_bert = BertForMaskedLM.from_pretrained("bert-base-uncased")
#
# # Load the RoBERTa tokenizer and model
# tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
# model_roberta = RobertaForMaskedLM.from_pretrained("roberta-base")
#
# model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
# tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
def accuracy(results):
    predicted_labels = []
    for item in results:
        if item[0] > item[1] or item[0] > item[2]:  # (true > false) and (true > unknown)
            predicted_labels.append(1)  # append 0 for entailment
        elif item[2] > item[0] and item[2] > item[1]:  # (unknown > true) and (unknown > false)
            predicted_labels.append(1)  # append 1 for neutral
        else:
            predicted_labels.append(2)  # else append 2 for contradiction

    print("The accuracy score is: {:.4f}".format(accuracy_score(all_labels, predicted_labels)))


if __name__ == '__main__':
    my_template_path = 'templates/template0/template_0_valid.json'

    dataset_v2 = load_json(my_template_path)
    my_custom_dataset = MyDataset(dataset_v2)
    my_dataloader = DataLoader(my_custom_dataset, batch_size=8, shuffle=False)
    all_labels = [entry['label'] for entry in dataset_v2]

    main.plot_distribution(all_labels)

    tokenizer_bert, model_bert = load_model("bert-base-uncased")
    tokenizer_roberta, model_roberta = load_model("roberta-base")
    tokenizer_gpt2, model_gpt2 = load_model("gpt2")

    models = [model_bert, model_roberta, model_gpt2]
    tokenizers = [tokenizer_bert, tokenizer_roberta, tokenizer_gpt2]
    probs_of_models_over_dateset = []
    for i, (model, tokenizer) in enumerate(zip(models, tokenizers)):
        probs_over_dateset = main.prob_distribution_over_vocab_with_batch(model, tokenizer, my_dataloader)
        # probs_of_models_over_dateset.append(probs_over_dateset)
        print("Finished model {}: {} \n".format(i + 1, model.name_or_path.split("-")[0]))
        results = [inner_list for outer_list in probs_over_dateset for inner_list in outer_list]
        accuracy(results)
    #
    # results_bert_flattened = [inner_list for outer_list in probs_of_models_over_dateset[0] for inner_list in outer_list]
    # results_roberta_flattened = [inner_list for outer_list in probs_of_models_over_dateset[1] for inner_list in outer_list]
    # results_gpt2_flattened = [inner_list for outer_list in probs_of_models_over_dateset[2] for inner_list in outer_list]
    #
    # accuracy(results_bert_flattened)
    # accuracy(results_roberta_flattened)
    # accuracy(results_gpt2_flattened)
