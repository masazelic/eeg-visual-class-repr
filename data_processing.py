import torch 

def load_data(filepath):
    """ Loading data from the filepath. """
    
    data = torch.load(filepath)
    
    # Data is dictionary with components 'dataset', 'labels', 'images'
    # We need 'dataset'
    return data['dataset']

def remove_33(dataset):
    """ Removes label 33 from dictionary. """
    
    new_dataset = [element for element in dataset if element['label'] != 33]
    
    return new_dataset

def cut_to_440_samples_and_std(dataset):
    """ Standardizes length of all signals to 440 samples by removing first 20 samples and using remaining 440. """
    
    for i in range(len(dataset)):
        dataset[i]['eeg'] = dataset[i]['eeg'][:, 20:20+440]
        
        mean = dataset[i]['eeg'].mean(dim=1, keepdims=True)
        std = dataset[i]['eeg'].std(dim=1, unbiased=False, keepdims=True)
        
        dataset[i]['eeg'] = (dataset[i]['eeg'] - mean) / std
    
    return dataset

def min_max_scaling(dataset):
    """ To get CNN embeddings, one step is min-max normalization. """
    
    for i in range(len(dataset)):
        
        min_vals = dataset[i]['eeg'].min(dim=1, keepdim=True)[0]
        max_vals = dataset[i]['eeg'].max(dim=1, keepdim=True)[0]
        
        dataset[i]['eeg'] = (dataset[i]['eeg']) / (max_vals - min_vals + 1e-8)
        
    return dataset

def train_val_test_split(dataset):
    """ Train-val-test split. """
    
    # Extract all the labels that are in the dataset
    all_labels = []
    for element in dataset:
        all_labels.append(element['label'])
        
    all_labels_unique = set(all_labels)
    
    # Get labels of images per each class
    all_images = []
    for label in all_labels_unique:
        images = []
        for element in dataset:
            if element['label'] == label:
                images.append(element['image'])
        all_images.append({'label': label, 'images': images})
        
    # Get train-val-test split
    train_data = []
    val_data = []
    test_data = []
    
    for entry in all_images:
        
        label = entry['label']
        images = entry['images']
        
        # Split the images in train-val-test split
        images = list(images)
        train_idx = images[:int(len(images) * 0.7)]
        val_idx = images[int(len(images) * 0.7):int(len(images) * 0.85)]
        test_idx = images[int(len(images) * 0.85):]
    
        for element in dataset:
            if element['label'] == label:
                if element['image'] in train_idx:
                    train_data.append(element)
                elif element['image'] in val_idx:
                    val_data.append(element)
                else:
                    test_data.append(element)

    return train_data, val_data, test_data
    