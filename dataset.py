from clip_classifier import load_cod10k_lazy
from torchvision import transforms
from torch.utils.data import DataLoader

tensor_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
LABEL_MAPPER = None

def pytorch_transform_fn(examples):
    # examples is a dict of lists: {'image': [PIL, ...], 'label': [0, ...]}
    examples['pixel_values'] = [tensor_transform(img.convert("RGB")) for img in examples['image']]
    del examples['image'] # ensures we only have 2 cols for loader
    return examples

def create_loader(batch_size=4, num_workers=1, split_name = 'train'):
    global LABEL_MAPPER
    dataset = load_cod10k_lazy()[split_name]
    LABEL_MAPPER = dataset.features['label']
    dataset.set_transform(pytorch_transform_fn)
    loader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle = (split_name == 'train'),
        num_workers = num_workers 
    )
    return loader

def label_int2str(label_int : int):
    return LABEL_MAPPER.int2str(label_int)
    

if __name__ == "__main__":
    loader = create_loader(4, 1, 'train')
    
    for item in loader: # dict{'label', 'pixel_values'}
        img_tensor = item['pixel_values']
        label_int = item['label']
        print("Image tensor:", img_tensor.shape)
        print("Labels:", label_int)
        print("Str labels:", label_int2str(label_int))
        break
