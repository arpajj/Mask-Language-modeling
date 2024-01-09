from datasets import load_dataset
import json
import os

def create_template(base_sentence, dataset):
    new_dataset = []
    for item in dataset:
        template = base_sentence.format(item['sentence_A'], item['sentence_B'])
        sample = {"sentence": template,
                  "premises": item['sentence_A'],
                  "hypothesis": item['sentence_B'],
                  "label": item['label']}
        new_dataset.append(sample)
    return new_dataset

def save_template(path, text):
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(text, json_file, ensure_ascii=False, indent=4)
    print("Saving complete!")

if __name__ == '__main__':
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
    template_dic = [
        "If {} is 'true', the {} is '[MASK]'",
        "Suppose that {} is: true, then {} is: [MASK]",
        "When {} is true, then {} is [MASK]",
        "When the premise '{}' is: true, then the hypothesis '{}' is: [MASK]",
        "When the premise {} is true, then the hypothesis {} is: [MASK]"
        "Considering that the premise '{}' is 'true', then the hypothesis '{}' is: '[MASK]'"
        "Knowing that the premise '{}' has a label 'true', then the hypothesis '{}' will have a label '[MASK]'"
        "Regarding that the premise '{}' is: 'true', then the hypothesis '{}' will be '[MASK]'"
    ]

    template_num = 0

    base_sentence = template_dic[template_num]

    # Create the template in the right form
    dataset_train = create_template(base_sentence, train_data)
    dataset_valid = create_template(base_sentence, validation_data)
    dataset_test = create_template(base_sentence, test_data)
    # Specify the file path where you want to save the JSON file
    dir_path = "templates/template{}".format(template_num)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    save_template('templates/template{}/template_{}_train.json'.format(template_num, template_num), dataset_train)
    save_template('templates/template{}/template_{}_valid.json'.format(template_num, template_num), dataset_valid)
    save_template('templates/template{}/template_{}_test.json'.format(template_num, template_num), dataset_test)
