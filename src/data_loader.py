import os
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def find_correct_root(start_path):
    for root, dirs, files in os.walk(start_path):
        if len(dirs) > 1:
            return root
    return start_path

def summarize_dataset(root_path):
    data = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                label = os.path.basename(root)
                full_path = os.path.join(root, file)
                data.append({'path': full_path, 'label': label})

    df = pd.DataFrame(data)
    
    stats = df['label'].value_counts().reset_index()
    stats.columns = ['Cell Type', 'Count']
    
    return df, stats

def get_dataloaders(data_dir, batch_size=32, img_size=128):

    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Normalization
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    return loader, dataset.class_to_idx