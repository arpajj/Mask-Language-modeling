import json
from torch.utils.data import DataLoader, Dataset

def load_json(path):
    with open(path, 'r') as json_file:
        dataset = json.load(json_file)
    return dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    my_template_path = 'templates/template2.json'

    dataset_v2 = load_json(my_template_path)

    # Create the Dataset object and the DataLoader
    my_custom_dataset = MyDataset(dataset_v2)
    my_dataloader = DataLoader(my_custom_dataset, batch_size=8, shuffle=False)
    all_labels = [entry['label'] for entry in dataset_v2]