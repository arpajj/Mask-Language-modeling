import json, pickle 
from torch.utils.data import DataLoader, Dataset
#from create_templates import all_datasets

project_path = "C:/Users/admitos/Desktop/Logic and Language/Project/templates/"
with open(project_path+'others/all_datasets.pkl', 'rb') as file:
    all_datasets_loaded = pickle.load(file)

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


## Create all the datatloaders
all_dataloaders = []
for i, my_dt in enumerate(all_datasets_loaded):
    dataloaders, labels_total = make_dataloader(my_dt, i)
    if i == 0:
        labels_final = labels_total

    all_dataloaders.append(dataloaders)
            
print("Finished with DataLoaders creation!!!")

##### OLD CODE #####
# Create the Dataset object and the DataLoader
# my_custom_dataset = MyDataset(dataset_v2)
# my_dataloader = DataLoader(my_custom_dataset, batch_size=8, shuffle=False)
# all_labels = [entry['label'] for entry in dataset_v2]

