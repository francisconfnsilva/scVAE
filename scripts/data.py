import kagglehub
import os
import shutil

def main():
    print("Downloading dataset from Kaggle...")
    cache_path = kagglehub.dataset_download("unclesamulus/blood-cells-image-dataset")
    
    target_path = os.path.abspath("./data")
    
    if os.path.exists(target_path):
        if os.path.islink(target_path) or not os.listdir(target_path):
            os.remove(target_path) if os.path.islink(target_path) else shutil.rmtree(target_path)

    print(f"Copying files to {target_path}...")
    shutil.copytree(cache_path, target_path) 
    print("Data successfully localized to ./data")

if __name__ == "__main__":
    main()