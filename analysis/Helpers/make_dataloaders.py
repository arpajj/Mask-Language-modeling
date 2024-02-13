import math, pickle 
from torch.utils.data import DataLoader, Dataset

average_length_test_sets_snli = [27.42, 28.42, 27.42, 31.42, 33.42, 32.42, 37.42, 33.42, 23.42, 23.42]
average_length_test_sets_sick = [25.24, 26.24, 25.24, 29.24, 31.24, 30.24, 35.24, 31.24, 21.24, 21.24]
average_length_test_sets_mnli = [36.7, 37.7, 36.7, 40.7, 42.7, 41.7, 46.7, 42.7, 32.53, 32.53]

# Selecte here the path that you have stored the templates
project_path = "C:/Users/admitos/Desktop/LoLa/Project/templates/"
with open(project_path+'others/all_datasets.pkl', 'rb') as file:
    all_datasets_loaded = pickle.load(file)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def short_long_sentences(a_dataset, template_num, name):
    short_sentences = []
    long_sentences = []
    if name.lower() == "snli": 
        average_length_test_sets = average_length_test_sets_snli
    elif name.lower() == "multi_nli": 
        average_length_test_sets = average_length_test_sets_mnli
    else: 
        average_length_test_sets = average_length_test_sets_sick    
    for item in a_dataset:
        if len(item['sentence'].split()) <= math.floor(average_length_test_sets[template_num]):
            short_sentences.append(item)
        else:
            long_sentences.append(item)
        
    print("The template is: {} and the Short and Long items are respectively: {} and {}".format(template_num, len(short_sentences), len(long_sentences)))
    return(short_sentences, long_sentences)

# Create the Dataset object and the DataLoader
def make_dataloader(datasets, temp_num, d_name):
    my_dataloaders = []
    all_labels = []
    for j, dtst in enumerate(datasets):
        my_custom_dataset = MyDataset(dtst)
        filtered_dataset = [structure for structure in my_custom_dataset if structure.get('label') != -1]
        if (j==1): # only for the test set we are going to distinct what sentences are short or long
            short, long = short_long_sentences(filtered_dataset, temp_num, d_name)
        my_dataloader = DataLoader(filtered_dataset, batch_size=8, shuffle=False)
        my_dataloaders.append(my_dataloader)
        if temp_num == 0:
            labels = [entry['label'] for entry in dtst if entry['label'] != -1] # exclude -1 form the SNLI dataset
            all_labels.append(labels)
    return(my_dataloaders, all_labels, short, long)

## Create all the datatloaders
dname = input("Enter a dataset name: ")
all_dataloaders = []
all_short_dataloaders = []
all_long_dataloaders = []
for i, my_dt in enumerate(all_datasets_loaded):
    dataloaders, labels_total, short_data, long_data = make_dataloader(my_dt, i, dname)
    if i == 0:
        labels_final = labels_total
    all_dataloaders.append(dataloaders)
    all_short_dataloaders.append(DataLoader(short_data, batch_size=8, shuffle=False))
    all_long_dataloaders.append(DataLoader(long_data, batch_size=8, shuffle=False))


print("Finished with all DataLoaders creation!!!")

