import torch, random
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np 

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

def roberta_mask_token_id(tokenizer, tokenized_input, word_tokens):
    sim_tok = [tok for tok in word_tokens if '_' in tok]
    mask_position = word_tokens.index(sim_tok[0])

    tokenized_input_list = tokenized_input["input_ids"].flatten().tolist()
    attention_mask_list = tokenized_input["attention_mask"].flatten().tolist()

    tokenized_input_list[mask_position] = tokenizer.mask_token_id

    tokenized_input["input_ids"] = torch.tensor(tokenized_input_list).unsqueeze(dim=0)
    tokenized_input["attention_mask"] = torch.tensor(attention_mask_list).unsqueeze(dim=0)
    
    return(tokenized_input)

def get_probabilities_per_batch(outs, t_batch, tokenizer, tokens_of_interest, topk_probs):
    mask_token_indicies = []
    if tokenizer.name_or_path == "gpt2": 
        for item in t_batch["input_ids"]:
            word_tokens = tokenizer.convert_ids_to_tokens(item)
            sim_tok = [tok for tok in word_tokens if '_' in tok]
            mask_token_indicies.append(word_tokens.index(sim_tok[0]))
            #mask_token_indicies.append(torch.where(item!=0)[0][-1].item())
    else:
        for item in t_batch["input_ids"]:
            mask_token_indicies.append(torch.where(item==tokenizer.mask_token_id)[0].item())

    extracted_rows = outs.logits[torch.arange(outs.logits.size(0)), mask_token_indicies]
    probabilities = F.softmax(extracted_rows, dim=-1)
    top_values, top_indices = torch.topk(probabilities, topk_probs, dim=1, largest=True, sorted=True)
    tokenizer_vocab = tokenizer.get_vocab()
    top_indices_list = top_indices.tolist()
    top_tokens = [[[token for token, idx in tokenizer_vocab.items() if idx == row] for row in top_indices_list[i]] for i in range(0,8)]
    inter_tokens_ids = get_interest_tokens(tokenizer.get_vocab(), tokens_of_interest)
    probabilites_of_interest = probabilities[:, inter_tokens_ids]
    return(probabilites_of_interest, top_tokens)


def mapping_dicts(target, replacer):
    for i in range(len(replacer)):
        target['input_ids'][i] = replacer[i]['input_ids'].flatten().tolist()
        target['attention_mask'][i] = replacer[i]['attention_mask'].flatten().tolist()
    return(target)


def prob_distribution_over_vocab_with_batch(model, tokenizer, my_dataloader, interest_tokens, top_tokens_return=100, valid=False):

    probs_over_batched_dataset = []
    top_tokens_per_batch = []
    for bi, batch in enumerate(my_dataloader):
        if bi%5 == 0: 
            print("On batch {} out of {} bathces".format(bi,len(my_dataloader)))
        if (model.name_or_path == 'roberta-base' or model.name_or_path == 'facebook/bart-base'): 
            if ("'[MASK]'" in batch['sentence'][0]): 
                sentences = [sentence.replace("'[MASK]'", "_") for sentence in batch['sentence']]
            else:
                sentences = [sentence.replace("[MASK]", "_") for sentence in batch['sentence']]
        elif (model.name_or_path=="gpt2"):
            if ("'[MASK]'" in batch['sentence'][0]): 
                sentences = [sentence.replace("'[MASK]'", "_") for sentence in batch['sentence']]
            else:
                sentences = [sentence.replace("[MASK]", "_") for sentence in batch['sentence']]
        else:
            sentences = batch['sentence']
        
        tokenized_batch = tokenizer(sentences)
        if (model.name_or_path == 'roberta-base' or model.name_or_path== 'facebook/bart-base'): 
            mykeys = list(tokenized_batch.keys())
            input_ids, attention_masks = list(tokenized_batch.values())
            correct_sents = []
            for inp, att in zip(input_ids, attention_masks):
                tokens = tokenizer.convert_ids_to_tokens(inp)
                one_tok_sent = {mykeys[0]: torch.tensor(inp), mykeys[1]: torch.tensor(att)}
                one_tok_sent = roberta_mask_token_id(tokenizer, one_tok_sent, tokens)
                correct_sents.append(one_tok_sent)
            
            tokenized_batch = mapping_dicts(tokenized_batch, correct_sents)
        tokenized_batch = transform_pad_batch(tokenized_batch, model)
        with torch.no_grad():
          outputs = model(**tokenized_batch)

        probs_of_interest, top_tokens = get_probabilities_per_batch(outputs, tokenized_batch, tokenizer, interest_tokens, top_tokens_return)
        probs_over_batched_dataset.append(probs_of_interest.tolist())
        top_tokens_per_batch.append(top_tokens)
        # Here you can put a break statement in order to run a the desired portion of the dataset. 
        # If 1% of the val dataset is desired (as the paper says) then 15 batches are fine. 
        if valid:
            if(bi>15): 
                break

    return(probs_over_batched_dataset,top_tokens_per_batch)


def plot_distribution(lbls, name):
    
    all_label_counts = Counter(lbls)    
    my_dict = {0: all_label_counts[0], 1: all_label_counts[1], 2: all_label_counts[2]}
    labels, counts = zip(*my_dict.items())
    bars = plt.bar(labels, counts, color =['green', 'blue', 'red'])
    plt.xlabel('Labels')
    plt.ylabel('Occurrences')
    plt.title('Distribution of Labels in ' + name)
    if(name.lower()=='snli' or name.lower()=='mnli'):
        plt.ylim(0, max(counts)+1250)
    plt.legend(bars, ['Entailment', 'Neutral', 'Contradiction'])
    plt.xticks(torch.arange(3))
    plt.grid(True)
    plt.show()


def get_distribution(lbls):
    all_label_counts = Counter(lbls)
    total = sum(all_label_counts.values())
    percentages = [round(100*val/total,2) for val in all_label_counts.values()]
    print("The distribution of labels is: \n Entailment: {}%, Neutral: {}%, Contradiction: {}%".format(percentages[1], percentages[0], percentages[2]))
    return(percentages)

def error_function(split):
    try:
        if split == "valid":
            ix = 0
        elif split == "test":
            ix = 1
        else:
            raise TypeError("You have given a wrong type of split.")
            
    except TypeError as e:
        print(f"An error occurred: {e}")
        ix = None
        
    return(ix)

def list_flattening(list_to_flatten):
    flattened_list = []
    for sublist in list_to_flatten:
        for element in sublist:
            flattened_list.append(element)
    return(flattened_list)

def average_tok_indicies(all_toks, my_itm):
    final_average_indices = []
    for class_toks in all_toks:
        sum_index = 0 
        for tok in class_toks: 
            if tok in my_itm:
                sum_index = sum_index + my_itm.index(tok)
            else: 
                sum_index = sum_index + len(my_itm)

        final_average_indices.append(sum_index/len(class_toks))
        
    return(final_average_indices)

def make_predictions(triplet):
    score_0, score_1, score_2 = triplet
    my_dictionary = {'0': score_0, '1': score_1, '2': score_2}
    min_score = min(score_0, score_1, score_2)

    if score_0 == score_1 == score_2:
        prediction = random.choice([0, 1, 2])
    else:
        lowest_labels = [label for label, score in my_dictionary.items() if score == min_score]
        prediction = random.choice(lowest_labels)

    return prediction

def find_accuracy_short_long(probabilities_of_models, my_test_labels):
    testing_models = ["BERT", "RoBERTa", "GPT2", "BART"]
    original_labels = my_test_labels
    results_bert_flattened = list_flattening(probabilities_of_models[0])
    results_roberta_flattened = list_flattening(probabilities_of_models[1])
    results_gpt2_flattened = list_flattening(probabilities_of_models[2])
    results_bart_flattened = list_flattening(probabilities_of_models[3])
    all_results_flattened = [results_bert_flattened, results_roberta_flattened, results_gpt2_flattened, results_bart_flattened]
    baseline_random = []
    all_predicted_labels = []
    for j, one_model_flat in enumerate(all_results_flattened):
        predicted_labels = []
        for item in one_model_flat:
            if (j==3):
                baseline_random.append(random.randint(0,2))
            positive = np.mean(item[0:3])
            negative = np.mean(item[3:6])
            neutral = np.mean(item[6:]) 

            if positive > negative and positive > neutral: # (true > false) and (true > unknown)
                predicted_labels.append(0) # append 0 for entailment 

            elif negative > positive and negative > neutral: # (false > true) and (false > unknown)
                predicted_labels.append(2) # append 2 for contradiction
            else: 
                predicted_labels.append(1) # else append 1 for neutral
        
        print("The predicted labels of {} are: {}".format(testing_models[j], predicted_labels))
        print("The accuracy score of {} is: {:.4f}".format(testing_models[j], accuracy_score(original_labels, predicted_labels)))
        all_predicted_labels.append(predicted_labels)
        print()
        if(j==3):
            print("The accuracy score the random method is: {:.4f}".format(accuracy_score(original_labels, baseline_random)))
    return(all_predicted_labels)

def find_accuracy_imp_tokens(probabilities_of_models, top_tokens_one_model, my_test_labels, my_mdl, all_interesting_tokens, lengths):
    original_labels = my_test_labels
    results_flattened = list_flattening(probabilities_of_models)
    top_tokens_flat = list_flattening(top_tokens_one_model)
    top_tokens_flat_x2 = [list_flattening(sublist) for sublist in top_tokens_flat]
    all_results_flattened = [results_flattened, top_tokens_flat_x2]
    all_predicted_labels = []
    for j, one_model_flat in enumerate(all_results_flattened):
        predicted_labels = []
        for item in one_model_flat:
            if j==0: 
                positive = np.mean(item[0:3])
                negative = np.mean(item[3:6])
                neutral = np.mean(item[6:]) 

                if positive > negative and positive > neutral: # (true > false) and (true > unknown)
                    predicted_labels.append(0) # append 0 for entailment 
                elif negative > positive and negative > neutral: # (false > true) and (false > unknown)
                    predicted_labels.append(2) # append 2 for contradiction
                else: 
                    predicted_labels.append(1) # else append 1 for neutral
            else: # j==1
                entail_toks = all_interesting_tokens[0:lengths[0]]
                contra_toks = all_interesting_tokens[lengths[0]:lengths[0]+lengths[1]]
                neutral_toks = all_interesting_tokens[lengths[0]+lengths[1]:]
                separated_toks = [entail_toks, contra_toks, neutral_toks]
                averages = average_tok_indicies(separated_toks, item)
                predicted_labels.append(int(make_predictions(averages)))
        if(j==0):
            print("The predicted labels of with the classical method are: {}".format(predicted_labels))
            print("The accuracy score of {} is: {:.4f}".format(my_mdl.name_or_path.split("-")[0].upper(),accuracy_score(original_labels, predicted_labels)))
            all_predicted_labels.append(predicted_labels)
        else: #j==1
            all_predicted_labels.append(predicted_labels)
            print("The predicted labels of with the new method are: {}".format(predicted_labels))
            print("The accuracy score the new method for {} is: {:.4f}".format(my_mdl.name_or_path.split("-")[0].upper(),accuracy_score(original_labels, predicted_labels)))

def untagle_tokens_based_on_labels(top_toks_seen, my_test_labels):

    def flattening_func(entail_top_toks,neutral_top_toks,contr_top_toks):
        entail_temp = list_flattening(entail_top_toks)
        neutral_temp = list_flattening(neutral_top_toks)
        contr_temp = list_flattening(contr_top_toks)

        entail_temp2 = list_flattening(entail_temp)
        neutral_temp2 = list_flattening(neutral_temp)
        contr_temp2 = list_flattening(contr_temp)

        entail_flattened = list_flattening(entail_temp2)
        neutral_flattened = list_flattening(neutral_temp2)
        contr_flattened = list_flattening(contr_temp2)

        return(entail_flattened,neutral_flattened,contr_flattened)

    neutral_top_tokens = []
    entail_top_tokens = []
    contr_top_tokens = []
    for k in range(len(top_toks_seen)):
        neutral_top_tokens_batch = []
        entail_top_tokens_batch = []
        contr_top_tokens_batch = []
        for idx, label in enumerate(my_test_labels[k:k+8]):
            if label == 1: 
                neutral_top_tokens_batch.append(top_toks_seen[k][idx])
            if label == 0:
                entail_top_tokens_batch.append(top_toks_seen[k][idx])
            if label == 2:
                contr_top_tokens_batch.append(top_toks_seen[k][idx])
        
        neutral_top_tokens.append(neutral_top_tokens_batch)
        entail_top_tokens.append(entail_top_tokens_batch)
        contr_top_tokens.append(contr_top_tokens_batch)

    batch = random.randint(0,len(top_toks_seen)-1)
    print("Choosen batch is: ", batch)
    print()
    print("Neutrality:")
    for item in neutral_top_tokens[batch]:
        print(item)

    print()
    print("Contradiction:")
    for item in contr_top_tokens[batch]:
        print(item)

    print()
    print("Entailment:")
    for item in entail_top_tokens[batch]:
        print(item)

    entail_flattened, neutral_flattened, contr_flattened = flattening_func(entail_top_tokens,neutral_top_tokens,contr_top_tokens)

    return(entail_flattened,neutral_flattened,contr_flattened) 

def make_ue_un_ue(en_flat, neut_flat, contr_flat):
    set1 = set(en_flat)
    set2 = set(neut_flat)
    set3 = set(contr_flat)

    common_elements = set1.intersection(set2, set3)
    common_elements_list = list(common_elements)
    print("Common tokens are:", common_elements_list)

    print()
    only_in_list1 = set1.difference(set2, set3)
    only_in_list2 = set2.difference(set1, set3)
    only_in_list3 = set3.difference(set1, set2)

    only_in_list1_list = list(only_in_list1)
    only_in_list2_list = list(only_in_list2)
    only_in_list3_list = list(only_in_list3)

    print("Tokens only in entailment:", only_in_list1_list)
    print("Tokens only in neutral:", only_in_list2_list)
    print("Tokens only in contradiction:", only_in_list3_list)
    print()
    print("The number of tokens only in entailment:", len(only_in_list1_list))
    print("The number of tokens only in neutral:", len(only_in_list2_list))
    print("The number of tokens only in contradiction:", len(only_in_list3_list))

