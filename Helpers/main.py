import torch 
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter


tokens_of_interest = ["true", "false", "unknown"]

def transform_pad_batch(t_batch, model):
    temp_input_ids = [torch.tensor(item) for item in t_batch['input_ids']]
    t_batch['input_ids'] = pad_sequence(temp_input_ids, batch_first=True, padding_value=0)
    if model.name_or_path == 'bert-base-uncased':
        temp_token_type_ids = [torch.tensor(item) for item in t_batch['token_type_ids']]
        t_batch['token_type_ids'] = pad_sequence(temp_token_type_ids, batch_first=True, padding_value=0)
    temp_attention_mask = [torch.tensor(item) for item in t_batch['attention_mask']]
    t_batch['attention_mask'] = pad_sequence(temp_attention_mask, batch_first=True, padding_value=0)
    return(t_batch)


def get_interest_tokens(vocab, tokens_interest):
    token_ids_of_interest = [vocab[token] for token in tokens_interest]
    return(token_ids_of_interest)


def roberta_mask_token_id(tokenizer, tokenized_input):
    # Manually set a position to be the [MASK] token --> It should be the penultimate position of the tokenized sentence.
    mask_position = len(tokenized_input["input_ids"].flatten()) - 1

    tokenized_input_list = tokenized_input["input_ids"].flatten().tolist()
    attention_mask_list = tokenized_input["attention_mask"].flatten().tolist()

    tokenized_input_list.insert(mask_position, tokenizer.mask_token_id)
    attention_mask_list.insert(0,1)

    tokenized_input["input_ids"] = torch.tensor(tokenized_input_list).unsqueeze(dim=0)
    tokenized_input["attention_mask"] = torch.tensor(attention_mask_list).unsqueeze(dim=0)
    
    return(tokenized_input)

def get_probabilities_per_batch(outs, t_batch, tokenizer):
    mask_token_indicies = []
    if tokenizer.name_or_path == "gpt2": 
        for item in t_batch["input_ids"]:
            mask_token_indicies.append(torch.where(item!=0)[0][-1].item())
    else:
        for item in t_batch["input_ids"]:
            mask_token_indicies.append(torch.where(item==tokenizer.mask_token_id)[0].item())

    extracted_rows = outs.logits[torch.arange(outs.logits.size(0)), mask_token_indicies]
    probabilities = F.softmax(extracted_rows, dim=-1)
    inter_tokens_ids = get_interest_tokens(tokenizer.get_vocab(), tokens_of_interest)
    probabilites_of_interest = probabilities[:, inter_tokens_ids]
    return(probabilites_of_interest)


def mapping_dicts(target, replacer):
    for i in range(len(replacer)):
        target['input_ids'][i] = replacer[i]['input_ids'].flatten().tolist()
        target['attention_mask'][i] = replacer[i]['attention_mask'].flatten().tolist()
    return(target)


def prob_distribution_over_vocab_with_batch(model, tokenizer, my_dataloader):

    probs_over_batched_dataset = []
    for bi, batch in enumerate(my_dataloader):
        if (model.name_or_path == 'roberta-base'): 
            sentences = [sentence[:-len('[MASK]')].strip() for sentence in batch['sentence']]
        elif (model.name_or_path=="gpt2"):
            sentences = [sentence.replace("[MASK]", '_') for sentence in batch['sentence']]
        else:
            sentences = batch['sentence']
        
        tokenized_batch = tokenizer(sentences)
        if (model.name_or_path == 'roberta-base'): 
            mykeys = list(tokenized_batch.keys())
            input_ids, attention_masks = list(tokenized_batch.values())
            correct_sents = []
            for inp, att in zip(input_ids, attention_masks):
                one_tok_sent = {mykeys[0]: torch.tensor(inp), mykeys[1]: torch.tensor(att)}
                one_tok_sent = roberta_mask_token_id(tokenizer, one_tok_sent)
                correct_sents.append(one_tok_sent)
            
            tokenized_batch = mapping_dicts(tokenized_batch, correct_sents)
        tokenized_batch = transform_pad_batch(tokenized_batch, model)
        with torch.no_grad():
          outputs = model(**tokenized_batch)

        probs_of_interest = get_probabilities_per_batch(outputs, tokenized_batch, tokenizer)
        probs_over_batched_dataset.append(probs_of_interest.tolist())

    return(probs_over_batched_dataset)

def plot_distribution(lbls):
    all_label_counts = Counter(lbls)
    for label, count in all_label_counts.items():
        print(f"Label {label}: {count} occurrences")

    labels, counts = zip(*all_label_counts.items())
    bars = plt.bar(labels, counts, color =['green', 'blue', 'red'])
    plt.xlabel('Labels')
    plt.ylabel('Occurrences')
    plt.title('Distribution of Labels')
    plt.legend(bars, ['Neutral', 'Entailment', 'Contradiction'])
    plt.xticks(torch.arange(3))
    plt.grid(True)
    plt.show()

def get_distribution(lbls):
    all_label_counts = Counter(lbls)
    total = sum(all_label_counts.values())
    percentages = [round(100*val/total,2) for val in all_label_counts.values()]
    print("The distribution of labels is: \n Entailment: {}%, Neutral: {}%, Contradiction: {}%".format(percentages[1], percentages[0], percentages[2]))
    return(percentages)

