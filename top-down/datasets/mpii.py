import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as transforms

class MPIIDataset(Dataset):
    """
    Dataloader for MPII Human Pose dataset in COCO format
    """
    def __init__(self,
                 target_size,
                 annotations_file='datasets/data/annotations/train.json', 
                 img_dir='datasets/data/',
                 transform=None):

        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)

        # Create id mappings
        self.img_id_to_img = {img['id']: img for img in self.coco_data['images']}

        # Create flattened samples list where each item is (image_id, annotation)
        self.samples = []
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id in self.img_id_to_img:  # Make sure image exists
                self.samples.append((img_id, ann))
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            
        # Get keypoint info
        self.num_keypoints = len(self.coco_data['categories'][0]['keypoints'])

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        img_id, ann = self.samples[idx]
        img_info = self.img_id_to_img[img_id]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
                
        # Process keypoints
        kps = np.array(ann['keypoints']).reshape(-1, 3)

        bbox = ann["bbox"]
        
        # Crop the image
        image = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            
        # Adjust keypoints relative to the cropped image
        kps[:, 0] = kps[:, 0] - bbox[0]
        kps[:, 1] = kps[:, 1] - bbox[1]

        # Transform image
        if self.transform:
            image = self.transform(image)

        # Calculate scale factors for keypoint coordinates
        scale_x = self.target_size[0] / bbox[2]
        scale_y = self.target_size[1] / bbox[3]
        
        # Scale keypoints
        scaled_kps = kps.copy()
        scaled_kps[:, 0] = scaled_kps[:, 0] * scale_x
        scaled_kps[:, 1] = scaled_kps[:, 1] * scale_y

        # Normalize keypoints
        scaled_kps[:, 0] /= bbox[2]
        scaled_kps[:, 1] /= bbox[3]

        # Convert to tensor
        keypoints = torch.tensor(scaled_kps, dtype=torch.float32)
        bbox = torch.tensor(bbox)

        return image, keypoints, bbox

def getMPIIDataloaders(config):
    train_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config["target_size"]),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MPIIDataset(target_size=config["target_size"], transform=train_transform)
    test_dataset = MPIIDataset(target_size=config["target_size"], annotations_file="datasets/data/annotations/train.json")

    indices = torch.randperm(len(train_dataset))
    val_size = len(train_dataset) // 8
    train_split = Subset(train_dataset, indices[:-val_size])
    val_split = Subset(train_dataset, indices[-val_size:])

    train_loader = DataLoader(
        train_split,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_split,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config["batch_size"])
    
    return train_loader, val_loader, test_loader
