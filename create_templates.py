from datasets import load_dataset
import json
import os

from tqdm import trange

def create_template(base_sentence, dataset, label):
    """
    :param base_sentence: [template, predict_label (e.g. true/false), is_reverse]
    :param dataset: data
    :param label: dataset_name
    Create dataset with templates
    """
    new_dataset = []
    for item in dataset:
        if label == "sick":
            premise = item['sentence_A'][0].lower() + item['sentence_A'][1:]
            hypothesis = item['sentence_B'][0].lower() + item['sentence_B'][1:]
        else:
            premise = item['premise'][0].lower() + item['premise'][1:-1]
            hypothesis = item['hypothesis'][0].lower() + item['hypothesis'][1:-1]
            if item['label'] == -1:
                continue
        if base_sentence[2]:
            template = base_sentence[0].format(hypothesis, premise)
        else:
            template = base_sentence[0].format(premise, hypothesis)
        sample = {"sentence": template,
                  "premises": premise,
                  "hypothesis": hypothesis,
                  "predict_label": base_sentence[1],
                  "label": item['label']}
        new_dataset.append(sample)
    return new_dataset


def save_template(path, text):
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(text, json_file, ensure_ascii=False, indent=4)
    print("Saving complete!")


def create_dataset(dataset_name, template_dic, template_num):
    """
    Create dataset with templates, splitting the dataset to (train/valid/test) set.
    """
    # Load the SICK dataset
    dataset = load_dataset(dataset_name)

    # Access the training, validation, and test splits
    # train_data = dataset["train"]
    validation_data = dataset["validation"]
    test_data = dataset["test"]

    base_sentence = template_dic[template_num]

    # Create the template in the right form
    # dataset_train = create_template(base_sentence, train_data, label=dataset_name)
    dataset_valid = create_template(base_sentence, validation_data, label=dataset_name)
    dataset_test = create_template(base_sentence, test_data, label=dataset_name)
    # Specify the file path where you want to save the JSON file

    if not os.path.exists(f"templates/{dataset_name}"):
        os.makedirs(f"templates/{dataset_name}")

    dir_path = "templates/{}/template{}".format(dataset_name, template_num + 1)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # save_template('templates/{}/template{}/template_train.json'.format(dataset_name, template_num + 1), dataset_train)
    save_template('templates/{}/template{}/template_valid.json'.format(dataset_name, template_num + 1), dataset_valid)
    save_template('templates/{}/template{}/template_test.json'.format(dataset_name, template_num + 1), dataset_test)


if __name__ == '__main__':
    # template_dic = [
    #     ["if {} is 'true', the {} is '[MASK]'", ["true", "false", "unknown"], False],
    #     ["Suppose that {} is: true, then {} is: [MASK]", [" true", " false", " unknown"], False],
    #     ["When {} is true, then {} is [MASK]", [" true", " false", " unknown"], False],
    #     ["When the premise '{}' is: true, then the hypothesis '{}' is: [MASK]", [" true", " false", " unknown"], False],
    #     ["When the premise {} is true, then the hypothesis {} is: [MASK]", [" true", " false", " unknown"], False],
    #     ["Considering that the premise '{}' is 'true', then the hypothesis '{}' is: '[MASK]'", ["true", "false", "unknown"], False],
    #     ["Knowing that the premise '{}' has a label 'true', then the hypothesis '{}' will have a label '[MASK]'", ["true", "false", "unknown"], False],
    #     ["Regarding that the premise '{}' is: 'true', then the hypothesis '{}' will be '[MASK]'", ["true", "false", "unknown"], False],
    #     ["{}: true, {}: [MASK]", [" true", " false", " unknown"], False],
    #     ["{}? [MASK], therefore {}", [" Yes", " No", " Maybe"], False],
    #     ["{}. [MASK], {}", [" Yes", " No", " Maybe"], False],
    #     ["{}? [MASK] {}", [" Yes", " No", " Maybe"], False],
    #     ["{}. [MASK] , no , {}", [" Alright", " Except", " Watch"], False],
    #     ["{}. [MASK] this time {}", [" Regardless", " Unless", " Fortunately"], False],
    # ]

    template_dic = [
        ["When {} is true, then {} is [MASK].", [" true", " false"], False],
        ["When {} is false, then {} is [MASK].", [" false", " true"], False],
        ["When {} is true, then {} is [MASK].", [" true", " false"], True],
        ["When {} is false, then {} is [MASK].", [" false", " true"], True],

        ["Suppose that {} is: true, then {} is: [MASK]", [" true", " false"], False],
        ["Suppose that {} is: false, then {} is: [MASK]", [" false", " true"], False],
        ["Suppose that {} is: true, then {} is: [MASK]", [" true", " false"], True],
        ["Suppose that {} is: false, then {} is: [MASK]", [" false", " true"], True],

        ["When {} is true, then {} is [MASK]", [" true", " false"], False],
        ["When {} is false, then {} is [MASK]", [" false", " true"], False],
        ["When {} is true, then {} is [MASK]", [" true", " false"], True],
        ["When {} is false, then {} is [MASK]", [" false", " true"], True],

        ["When the premise '{}' is: true, then the hypothesis '{}' is: [MASK]", [" true", " false"], False],
        ["When the premise '{}' is: false, then the hypothesis '{}' is: [MASK]", [" false", " true"], False],
        ["When the premise '{}' is: true, then the hypothesis '{}' is: [MASK]", [" true", " false"], True],
        ["When the premise '{}' is: false, then the hypothesis '{}' is: [MASK]", [" false", " true"], True],

        ["When the premise {} is true, then the hypothesis {} is: [MASK]", [" true", " false"], False],
        ["When the premise {} is false, then the hypothesis {} is: [MASK]", [" false", " true"], False],
        ["When the premise {} is true, then the hypothesis {} is: [MASK]", [" true", " false"], True],
        ["When the premise {} is false, then the hypothesis {} is: [MASK]", [" false", " true"], True],

        ["Considering that the premise '{}' is 'true', then the hypothesis '{}' is: '[MASK]'", ["true", "false"], False],
        ["Considering that the premise '{}' is 'false', then the hypothesis '{}' is: '[MASK]'", ["false", "true"], False],
        ["Considering that the premise '{}' is 'true', then the hypothesis '{}' is: '[MASK]'", ["true", "false"], True],
        ["Considering that the premise '{}' is 'false', then the hypothesis '{}' is: '[MASK]'", ["false", "true"], True],

        ["Knowing that the premise '{}' has a label 'true', then the hypothesis '{}' will have a label '[MASK]'.", ["true", "false"], False],
        ["Knowing that the premise '{}' has a label 'false', then the hypothesis '{}' will have a label '[MASK]'.", ["false", "true"], False],
        ["Knowing that the premise '{}' has a label 'true', then the hypothesis '{}' will have a label '[MASK]'.", ["true", "false"], True],
        ["Knowing that the premise '{}' has a label 'false', then the hypothesis '{}' will have a label '[MASK]'.", ["false", "true"], True],

        ["Regarding that the premise '{}' is: 'true', then the hypothesis '{}' will be '[MASK]'", ["true", "false"], False],
        ["Regarding that the premise '{}' is: 'false', then the hypothesis '{}' will be '[MASK]'", ["false", "true"], False],
        ["Regarding that the premise '{}' is: 'true', then the hypothesis '{}' will be '[MASK]'", ["true", "false"], True],
        ["Regarding that the premise '{}' is: 'false', then the hypothesis '{}' will be '[MASK]'", ["false", "true"], True],

    ]

    for i in trange(0, len(template_dic)):
        create_dataset("sick", template_dic, i)
        create_dataset("snli", template_dic, i)

