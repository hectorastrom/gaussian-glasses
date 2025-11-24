# @Time    : 2025-11-21 17:39
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : clip_classifier.py
import os
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Image as ImageFeature, ClassLabel, Value
import torch as t
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import pipeline

##################################
# Dataset loading
##################################

# [SCRIPT_DIR]/datasets/COD10K
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATASET_ROOT = os.path.join(SCRIPT_DIR, "datasets", "COD10K")

SPLITS = ["Train", "Test"]
# SubClass is located at index 5 in the filename split.
LABEL_INDEX = 5

def get_label_from_filename(filename: str) -> str:
    """Extracts the SubClass label from the COD10K filename."""
    parts = filename.split('-')
    
    if len(parts) <= LABEL_INDEX:
        # Get next best label at idx 3 - the superclass label
        if len(parts) <= 3:
            raise Exception(f"Malformed file name (too short): {filename}")
        else: 
            return parts[3]
    
    # Extract the SubClass name (e.g., 'Snake')
    return parts[LABEL_INDEX]

def image_label_generator(split_name: str):
    # Generators are memory efficient, lazy, and expected by datasets library
    """
    Generator function to yield (image, label_name) pairs.
    
    Note that images are stored as image_paths, and decoded by datasets.
    """
    # 'Train/Images' and 'Test/Images' structure
    split_path = os.path.join(DATASET_ROOT, split_name, "Image")
    
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Directory not found: {split_path}")

    for filename in sorted(os.listdir(split_path)):    
        if filename.endswith(".jpg"): # only jpg - i checked
            file_path = os.path.join(split_path, filename)
            label_name = get_label_from_filename(filename)
            
            yield {
                "image": file_path,
                "label_name": label_name
            }
        else:
            print(f"Not an image (harmless, skipped): {filename}")

def load_cod10k_lazy() -> DatasetDict:
    """
    Loads COD10K dataset with subclass as label. Lazy loading only loads
    filepaths, which are automatically decoded to images through datasets'
    ImageFeature() schema.
    
    Cols: 'image', 'label'
    """
    dataset_dict = {}
    
    # schema is info datasets need to load image on demand from filepath
    features_schema = Features({
        "image" : ImageFeature(),
        "label_name": Value('string')
    })
    
    for split in SPLITS:
        raw_dataset = Dataset.from_generator(
            image_label_generator,
            features=features_schema,
            gen_kwargs={"split_name": split} # this are kwargs it passes to image_label_generator
        )
        dataset_dict[split.lower()] = raw_dataset
        
    raw_datasets = DatasetDict(dataset_dict)
    
    # convert str class names to ClassLabel features
    all_labels = set()
    for split in raw_datasets:
        all_labels.update(raw_datasets[split]["label_name"])
        
    label_list = sorted(list(all_labels))
    # mapping of all possible labels to ints (categorical)
    class_feature = ClassLabel(names=label_list) 
    
    def encode_labels(sample):
        sample['label'] = class_feature.str2int(sample['label_name'])
        return sample
    
    print("Encoding labels...")
    raw_datasets = raw_datasets.map(encode_labels)
    
    # imgs remain untouched, and are decoded from filepaths JIT
    final_dataset = raw_datasets.map(
        encode_labels, 
        remove_columns=['label_name']
    )
    
    final_dataset = final_dataset.cast_column("label", class_feature)
    
    return final_dataset

if __name__ == "__main__":
    hf_dataset = load_cod10k_lazy()
    print("Success! Dataset loaded")

    ##################################
    # Rendering images from dset
    ##################################
    clip = pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        dtype=t.bfloat16,
        device=0,
        use_fast=True
    )

    class_features = hf_dataset['train'].features['label'] # ClassLabel mapper
    candidate_labels = [f"An image of {label}" for label in class_features.names]
    BATCH_SIZE = 16
    top_n = 1 # mark prediction within top_n of real label as correct

    total = 0
    correct = 0

    print(f"Beginning evaluation of {len(hf_dataset['train'])} images...")
    for i in range(0, len(hf_dataset["train"]), BATCH_SIZE):
        batch_raw = hf_dataset["train"][i : i + BATCH_SIZE]
        images = batch_raw['image'] # list of PIL images
        labels = batch_raw['label'] # list of int labels
        str_labels = class_features.int2str(labels) # list of str
        total += len(images)

        # handles batching internally
        result = clip(images, candidate_labels) # B x C list of list of dicts

        # evaluate accuracy
        for i, scores in enumerate(result):
            for candidate in scores[:top_n]:
                label_prediction = candidate['label'].split(' ')[-1]
                actual = str_labels[i]
                # print(f"One of our predictions: {label_prediction}")
                # print(f"actual label: {actual}")
                # print()
                if label_prediction == actual:
                    correct += 1
                break

    print(f"Correct: {correct}")
    print(f"Top_{top_n} accuracy: {correct / total}")
    
    # RESULT:
    # Correct: 1757
    # Top_1 accuracy: 0.29283333333333333
