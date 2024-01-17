from datasets import load_dataset
import json
import os
from torch.utils.data import DataLoader, Dataset
import pickle 
### -------  Here you can choose a template and formulate the base sentence as you please ------- ###
####   base_sentence = "if [P] is true, the [H] is [MASK]"

template_dic = [
    "If '{}' is 'true', the '{}' is '[MASK]'",
    "Suppose that '{}' is: true, then '{}' is: [MASK]",
    "When '{}' is true, then '{}' is [MASK]",
    "When the premise '{}' is: true, then the hypothesis '{}' is: [MASK]",
    "When the premise '{}' is true, then the hypothesis '{}' is: [MASK]",
    "Considering that the premise '{}' is 'true', then the hypothesis '{}' is: '[MASK]'",
    "Knowing that the premise '{}' has a label 'true', then the hypothesis '{}' will have a label '[MASK]'",
    "Regarding that the premise '{}' is: 'true', then the hypothesis '{}' will be '[MASK]'",
    "{}? || [MASK], {}",
    "'{}'? || [MASK], '{}'"
]

# Create templates
def create_template(base_sentence, dataset):
    new_dataset = []
    for item in dataset:
        template = base_sentence.format(item['sentence_A'].lower(), item['sentence_B'].lower())
        sample = {"sentence": template,
                  "premise": item['sentence_A'],
                  "hypothesis": item['sentence_B'],
                  "label": item['label']}
        new_dataset.append(sample)
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

# Create the Dataset object and the DataLoader
def make_dataloader(datasets, temp_num):
    my_dataloaders = []
    all_labels = []
    for dtst in datasets:
        my_custom_dataset = MyDataset(dtst)
        my_dataloader = DataLoader(my_custom_dataset, batch_size=8, shuffle=False)
        my_dataloaders.append(my_dataloader)
        if temp_num == 0:
            labels = [entry['label'] for entry in dtst]
            all_labels.append(labels)
    
    return(my_dataloaders, all_labels)


if __name__ == '__main__':
    # Load the SICK dataset
    sick_dataset = load_dataset("sick")

    # Access the training, validation, and test splits
    train_data = sick_dataset["train"]
    validation_data = sick_dataset["validation"]
    test_data = sick_dataset["test"]

    all_datasets = []
    project_path = "C:/Users/admitos/Desktop/Logic and Language/Project/templates/"
    for TEMPLATE_NUM in range(0,10):
        print("We are on template:", TEMPLATE_NUM+1)
        base_sentence = template_dic[TEMPLATE_NUM]

        dataset_train = create_template(base_sentence, train_data)
        dataset_valid = create_template(base_sentence, validation_data)
        dataset_test = create_template(base_sentence, test_data)

        save_template((project_path+'template{}/template_{}_train.json').format(TEMPLATE_NUM+1, TEMPLATE_NUM+1), dataset_train)
        save_template((project_path+'template{}/template_{}_valid.json').format(TEMPLATE_NUM+1, TEMPLATE_NUM+1), dataset_valid)
        save_template((project_path+'template{}/template_{}_test.json').format(TEMPLATE_NUM+1, TEMPLATE_NUM+1), dataset_test)

        ## Create the dataloaders 
        my_dataset = [dataset_train, dataset_valid, dataset_test]
        #dataloaders, labels_total = make_dataloader(my_dataset, TEMPLATE_NUM)
        # if TEMPLATE_NUM == 0:
        #     labels_final = labels_total
            
        all_datasets.append(my_dataset)
        print()

    print("Finished with templates creation!!!")

#save_template(project_path+'others/all_labels.json', labels_final)

#Save to pickle file
with open(project_path+'others/all_datasets.pkl', 'wb') as file:
     pickle.dump(all_datasets, file)
print("Saving all Datasets complete!")

# ################################################# OLD CODE  ########################################################

# premises = [item['sentence_A'].lower() for item in train_data] + [item['sentence_A'].lower() for item in validation_data] + [item['sentence_A'].lower() for item in test_data]
# hypotheses = [item['sentence_B'].lower() for item in train_data] + [item['sentence_B'].lower() for item in validation_data] + [item['sentence_B'].lower() for item in test_data]
# labels = [item['label'] for item in train_data] + [item['label'] for item in validation_data] + [item['label'] for item in test_data]

# # Create the template in the right form
# my_tempalte = [{"sentence": i1, "label": i2} for i1, i2 in zip([base_sentence.format(prem,hypo) for prem,hypo in zip(premises,hypotheses)],[mylbl for mylbl in labels])]

# # Specify the file path where you want to save the JSON file
# my_template_path = "C:/Users/admitos/Desktop/Logic and Language/Project/templates/template.json"

# with open(my_template_path, 'w', encoding='utf-8') as json_file:
#     json.dump(my_tempalte, json_file, ensure_ascii=False, indent=4)

