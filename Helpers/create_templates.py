from datasets import load_dataset
import json

# Load the SICK dataset
sick_dataset = load_dataset("sick")

# Access the training, validation, and test splits
train_data = sick_dataset["train"]
validation_data = sick_dataset["validation"]
test_data = sick_dataset["test"]

# # Display some information about the dataset
# print("Number of training examples:", len(train_data))
# print("Number of validation examples:", len(validation_data))
# print("Number of test examples:", len(test_data))
# print()
# # Accessing a sample pair from the training set
# for key, value in test_data[0].items():
#     print(key, ": ", value)



### -------  Here you can choose a template and formulate the base sentence as you please ------- ###
#base_sentence = "if [P] is true, the [H] is [MASK]"
base_sentence = "When the premise {} is true, then the hypothesis {} is [MASK]"

premises = [item['sentence_A'].lower() for item in train_data] + [item['sentence_A'].lower() for item in validation_data] + [item['sentence_A'].lower() for item in test_data]
hypotheses = [item['sentence_B'].lower() for item in train_data] + [item['sentence_B'].lower() for item in validation_data] + [item['sentence_B'].lower() for item in test_data]
labels = [item['label'] for item in train_data] + [item['label'] for item in validation_data] + [item['label'] for item in test_data]

# Create the template in the right form
my_tempalte = [{"sentence": i1, "label": i2} for i1, i2 in zip([base_sentence.format(prem,hypo) for prem,hypo in zip(premises,hypotheses)],[mylbl for mylbl in labels])]

# Specify the file path where you want to save the JSON file
my_template_path = 'C:/Users/admitos/Desktop/Logic and Language/Project/template.json'

with open(my_template_path, 'w', encoding='utf-8') as json_file:
    json.dump(my_tempalte, json_file, ensure_ascii=False, indent=4)
