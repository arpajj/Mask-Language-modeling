import json
from torch.utils.data import DataLoader, Dataset

my_template_path = 'C:/Users/admitos/Desktop/Logic and Language/Project/template2.json'

with open(my_template_path, 'r') as json_file:
    dataset_v2 = json.load(json_file)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create the Dataset object and the DataLoader
my_custom_dataset = MyDataset(dataset_v2)
my_dataloader = DataLoader(my_custom_dataset, batch_size=8, shuffle=False)
all_labels = [entry['label'] for entry in dataset_v2]

