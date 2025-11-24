import os
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, Features, Image as ImageFeature, ClassLabel, Value

##################################
# Dataset Reading
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

##################################
# Transforming to torch dataset
##################################

tensor_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def pytorch_transform_fn(examples):
    # examples is a dict of lists: {'image': [PIL, ...], 'label': [0, ...]}
    examples['pixel_values'] = [tensor_transform(img.convert("RGB")) for img in examples['image']]
    del examples['image'] # ensures we only have 2 cols for loader
    return examples

def build_COD_torch_dataset(split_name = 'train'):
    dataset = load_cod10k_lazy()[split_name]
    dataset.set_transform(pytorch_transform_fn)
    
    label_feature = dataset.features['label']
    # Adding some metadata that's helpful
    dataset.all_classes = label_feature.names 
    dataset.label2str = label_feature.int2str
    
    return dataset
    
if __name__ == "__main__":
    dataset = build_COD_torch_dataset('train')
    loader = DataLoader(dataset, batch_size=4, num_workers=1, shuffle=True)
    
    for item in loader: # dict{'label', 'pixel_values'}
        img_tensor = item['pixel_values']
        label_int = item['label']
        print("Image tensor:", img_tensor.shape)
        print(img_tensor.min(), img_tensor.max()) # images are [0, 1]
        print("Labels:", label_int)
        print("Str labels:", dataset.label2str(label_int))
        break
