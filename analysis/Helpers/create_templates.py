from datasets import load_dataset
import json
import os
from torch.utils.data import DataLoader, Dataset
import pickle 
### -------  Here you can choose a template and formulate the base sentence as you please ------- ###
####   base_sentence = "if [P] is true, the [H] is [MASK]"

template_dict = [
    "If '{}' is 'true', the '{}' is '[MASK]'",
    "Suppose that '{}' is: true, then '{}' is: [MASK]",
    "When '{}' is true, then '{}' is [MASK]",
    "When the premise '{}' is: true, then the hypothesis '{}' is: [MASK]",
    "When the premise '{}' has the label 'true', then the hypothesis '{}' is: [MASK]",
    "Considering that the premise '{}' is 'false', then the hypothesis '{}' is: '[MASK]'",
    "Knowing that the premise '{}' has a label 'false', then the hypothesis '{}' will have a label '[MASK]'",
    "Regarding that the premise '{}' is: 'false', then the hypothesis '{}' will be '[MASK]'",
    "{}? || [MASK], {}",
    "'{}'? || [MASK], '{}'"
]

# Create templates
def create_template(base_sentence, dataset, dataset_name):
    new_dataset = []
    if dataset_name == 'snli' or dataset_name == 'multi_nli':
        for item in dataset:
            template = base_sentence.format(item['premise'].lower(), item['hypothesis'].lower())
            sample = {"sentence": template,
                    "premise": item['premise'],
                    "hypothesis": item['hypothesis'],
                    "label": item['label']}
            new_dataset.append(sample)
    elif dataset_name == 'sick':
        for item in dataset:
            template = base_sentence.format(item['sentence_A'].lower(), item['sentence_B'].lower())
            sample = {"sentence": template,
                    "premise": item['sentence_A'],
                    "hypothesis": item['sentence_B'],
                    "label": item['label']}
            new_dataset.append(sample)
    else:
        print("Dataset error!!")
        exit()

    return new_dataset

# Save the template locally
def save_template(path, file):
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(file, json_file, ensure_ascii=False, indent=4)
    print("Saving complete!")

## Create Dataloaders 
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    # Load the SICK/SNLI/MNLI dataset
    # Here put the name of the dataset that you want to investigate: possible names: "sick" or "snli" or "multi_nli"
    print("You must choose one of the following possible dataset names: 'sick', 'snli', 'multi_nli'.")
    dname = input("Enter a dataset name: ")
    print("You entered:", dname)    
    
    dataset = load_dataset(dname)

    # Access validation and test set (Skip training set since we don't train)
    #train_data = sick_dataset["train"]
    if dname == 'multi_nli':
        validation_data = dataset['validation_matched']
        test_data = dataset['validation_mismatched']
    else:
        validation_data = dataset["validation"]
        test_data = dataset["test"]

    all_datasets = []
    # Select the path you want to store the templates
    project_path = "C:/Users/admitos/Desktop/LoLa/Project/templates/" 
    for TEMPLATE_NUM in range(0,10):
        print("We are on template:", TEMPLATE_NUM+1)
        base_sentence = template_dict[TEMPLATE_NUM]

        #dataset_train = create_template(base_sentence, train_data)
        dataset_valid = create_template(base_sentence, validation_data, dname)
        dataset_test = create_template(base_sentence, test_data, dname)

        #save_template((project_path+'template{}/snli_template_{}_train.json').format(TEMPLATE_NUM+1, TEMPLATE_NUM+1), dataset_train)
        save_template((project_path+'template{}/snli_template_{}_valid.json').format(TEMPLATE_NUM+1, TEMPLATE_NUM+1), dataset_valid)
        save_template((project_path+'template{}/snli_template_{}_test.json').format(TEMPLATE_NUM+1, TEMPLATE_NUM+1), dataset_test)

        ## Create the datasets 
        my_dataset =  [dataset_valid, dataset_test]
            
        all_datasets.append(my_dataset)
        print()

    print("Finished with templates creation!!!")

    #Save to pickle file
    # Select the path you want to store the datasets
    with open(project_path+'others/all_datasets.pkl', 'wb') as file:
        pickle.dump(all_datasets, file)
    print("Saving all Datasets complete!")
