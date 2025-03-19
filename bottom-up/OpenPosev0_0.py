import os
import cv2
import random
import wandb
import numpy as np
import scipy.io
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Global constants
IMAGE_SIZE = (256, 256)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
COLOR_JITTER_PARAMS = (0.2, 0.2, 0.2, 0.005)

# Define number of keypoints and a sample skeleton (limb connections)
NUM_KEYPOINTS = 16
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15)
]
NUM_LIMBS = len(SKELETON)

SIGMAS = np.array([
        .89,  # 0: r_ankle
        .87,  # 1: r_knee
        1.07, # 2: r_hip
        1.07, # 3: l_hip
        .87,  # 4: l_knee
        .89,  # 5: l_ankle
        1.07, # 6: pelvis
        1.07, # 7: thorax
        .87,  # 8: upper_neck
        .89,  # 9: head_top
        .62,  # 10: r_wrist # Changed from 0.79
        .72,  # 11: r_elbow
        .79,  # 12: r_shoulder
        .79,  # 13: l_shoulder
        .72,  # 14: l_elbow
        .62   # 15: l_wrist # Changed from 0.79
    ], dtype=np.float32) / 10.0 

run = wandb.init(
    entity = "renaudtp-oregon-state-university",
    project = "deep-learning-final-project",
    config = {
        "learning_rate": 1e-3,
        "architecture":"CNN",
        "dataset":"MPII",
        "epochs":50,
    },
)

# ----------------------- Helper Functions -----------------------

def compute_bbox(objpos, scale, factor=200):
    """
    Compute an approximate bounding box given an object center and scale.
    
    Args:
        objpos: array-like with [x_center, y_center] (in original image coordinates).
        scale: a scalar representing the scale of the person.
        factor: a constant multiplier to determine bbox size.
    
    Returns:
        bbox: a numpy array [x_min, y_min, width, height].
    """
    x_center, y_center = objpos
    width = scale * factor
    height = scale * factor
    x_min = x_center - width / 2.0
    y_min = y_center - height / 2.0
    return np.array([x_min, y_min, width, height], dtype=np.float32)


def getOKS(gt_keypoints, pred_keypoints, bbox, sigmas):
    """
    Compute the OKS (Object Keypoint Similarity) for a single instance.

    Args:
        gt_keypoints: numpy array of shape (N, 3) with (x, y, visibility).
        pred_keypoints: numpy array of shape (N, 2) with (x, y) predicted coords.
        bbox: numpy array [x_min, y_min, width, height] representing the bounding box.
        sigmas: numpy array of shape (N,) with per-keypoint constants (kappa_i in your equation).

    Returns:
        oks: float, the average OKS over visible keypoints.
    """
    # 1. Compute the object scale s from the bounding box. 
    #    Here, we define s as the sqrt of the box area, but adapt as needed.
    width = bbox[2]
    height = bbox[3]
    s = np.sqrt(width * height)  # or use another scale definition if you prefer

    # 2. Compute squared distances d_i^2 for each keypoint
    d2 = np.sum((gt_keypoints[:, :2] - pred_keypoints)**2, axis=1)

    # 3. Visibility mask
    vis = (gt_keypoints[:, 2] > 0)  # boolean array, True if keypoint is visible

    # 4. Compute the exponent for each visible keypoint:
    #    exp(- d_i^2 / (2 * s^2 * sigmas[i]^2))
    #    We'll store these in oks_vals.
    denom = 2 * (s**2) * (sigmas**2)
    oks_vals = np.exp(- d2 / denom)

    # 5. Zero out the contribution for non-visible keypoints
    oks_vals[~vis] = 0.0

    # 6. Average over visible keypoints
    visible_count = np.sum(vis)
    if visible_count > 0:
        return float(np.sum(oks_vals) / visible_count)
    else:
        return 0.0

# ----------------------- Utility Functions -----------------------
# Modify generate_heatmap to support a list of keypoints arrays:
def generate_heatmap(keypoints, output_size, sigma=2):
    """
    Generate per-keypoint Gaussian heatmaps.
    If keypoints is a list, compute the heatmap for each person and take the pixel-wise maximum.
    
    Args:
        keypoints: Either a numpy array (NUM_KEYPOINTS, 3) or a list of such arrays.
        output_size: tuple (height, width) for heatmap resolution.
        sigma: Gaussian sigma.
    
    Returns:
        heatmaps: numpy array (NUM_KEYPOINTS, H, W).
    """
    def single_heatmap(kp):
        hm = np.zeros((NUM_KEYPOINTS, output_size[0], output_size[1]), dtype=np.float32)
        for i in range(NUM_KEYPOINTS):
            x, y, v = kp[i]
            if v > 0:
                grid_y, grid_x = np.mgrid[0:output_size[0], 0:output_size[1]]
                heat = np.exp(-((grid_x - x)**2 + (grid_y - y)**2) / (2 * sigma**2))
                hm[i] = heat
        return hm

    # If keypoints is a list, aggregate heatmaps from all persons.
    if isinstance(keypoints, list):
        heatmaps_list = [single_heatmap(kp) for kp in keypoints]
        # Pixel-wise maximum over persons.
        aggregated_heatmaps = np.max(np.stack(heatmaps_list, axis=0), axis=0)
        return aggregated_heatmaps
    else:
        return single_heatmap(keypoints)


def generate_paf(keypoints, skeleton, output_size, limb_width=1):
    """
    Generate Part Affinity Fields (PAFs) for each limb.
    
    Args:
        keypoints: numpy array (NUM_KEYPOINTS, 3) with (x, y, visibility).
        skeleton: list of tuples defining limb connections.
        output_size: tuple (height, width) for PAF resolution.
        limb_width: threshold (in pixels) for PAF assignment.
    
    Returns:
        pafs: numpy array (2*NUM_LIMBS, H, W) with x and y channels for each limb.
    """
    num_limbs = len(skeleton)
    pafs = np.zeros((2 * num_limbs, output_size[0], output_size[1]), dtype=np.float32)
    for i, (j1, j2) in enumerate(skeleton):
        kp1 = keypoints[j1]
        kp2 = keypoints[j2]
        if kp1[2] > 0 and kp2[2] > 0:
            x1, y1 = kp1[:2]
            x2, y2 = kp2[:2]
            vec = np.array([x2 - x1, y2 - y1])
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            vec = vec / norm
            grid_y, grid_x = np.mgrid[0:output_size[0], 0:output_size[1]]
            d = np.abs((grid_x - x1) * vec[1] - (grid_y - y1) * vec[0])
            mask = d <= limb_width
            pafs[2*i][mask] = vec[0]
            pafs[2*i+1][mask] = vec[1]
    return pafs


def get_keypoints_from_heatmaps(heatmaps):
    """
    Decode keypoint coordinates from predicted heatmaps by taking the argmax.
    
    Args:
        heatmaps: torch.Tensor (batch, NUM_KEYPOINTS, H, W)
    
    Returns:
        keypoints_batch: list of numpy arrays (NUM_KEYPOINTS, 2) for each image.
    """
    keypoints_batch = []
    heatmaps_np = heatmaps.detach().cpu().numpy()
    for hm in heatmaps_np:
        keypoints = []
        for i in range(hm.shape[0]):
            idx = np.unravel_index(np.argmax(hm[i]), hm[i].shape)
            keypoints.append([idx[1], idx[0]])
        keypoints_batch.append(np.array(keypoints))
    return keypoints_batch


# ----------------------- Visualization Functions -----------------------
def visualize_predictions(image, predicted_keypoints, save_path=None):
    """
    Visualize predicted keypoints on the given image.
    
    Args:
        image: A PIL Image or torch.Tensor.
        predicted_keypoints: numpy array of shape (NUM_KEYPOINTS, 2).
        save_path: Optional path to save the visualization.
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        image = np.clip(image, 0, 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    for (x, y) in predicted_keypoints:
        plt.scatter(x, y, c='r', s=40)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# Update the visualization function for ground truth to plot all persons:
def visualize_ground_truth(image, keypoints, save_path=None):
    """
    Visualize rescaled ground truth keypoints on the given image.
    Supports keypoints as a single array (for one person) or a list (for multiple persons).
    
    Args:
        image: A PIL Image or torch.Tensor.
        keypoints: Either a numpy array of shape (NUM_KEYPOINTS, 3) or a list of such arrays.
        save_path: Optional path to save the visualization.
    """
    if isinstance(image, torch.Tensor):
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np = img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = np.array(image) / 255.0

    plt.figure(figsize=(6, 6))
    plt.imshow(img_np)
    
    # Define a list of colors to cycle through for different persons.
    colors = ['b', 'g', 'c', 'm', 'y', 'orange', 'purple']
    
    # If keypoints is a list, plot each person's keypoints.
    if isinstance(keypoints, list):
        for idx, kp in enumerate(keypoints):
            col = colors[idx % len(colors)]
            for (x, y, v) in kp:
                if v > 0:
                    plt.scatter(x, y, c=col, s=40)
    else:
        for (x, y, v) in keypoints:
            if v > 0:
                plt.scatter(x, y, c='b', s=40)
                
    plt.title("Rescaled Ground Truth Keypoints")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(2)
    plt.close()


# ----------------------- Dataset Definition -----------------------
# Update the __getitem__ method in MPIIDataset to collect all persons:
class MPIIDataset(Dataset):
    def __init__(self, mat_file, images_dir, transform=None, debug=False):
        """
        MPII HumanPose dataset loader that filters out images without valid keypoint annotations.
        
        Args:
            mat_file: Path to the MATLAB annotation file.
            images_dir: Directory containing the images.
            transform: torchvision transforms to apply.
            debug: If True, visualize rescaled ground truth keypoints.
        """
        super(MPIIDataset, self).__init__()
        self.images_dir = images_dir
        self.transform = transform
        self.debug = debug
        
        # Load the MATLAB file with attribute-style access.
        mat = scipy.io.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
        annolist = mat['RELEASE'].annolist
        
        total = len(annolist)
        valid_annolist = []
        excluded_count = 0
        
        for ann in annolist:
            file_name = ann.image.name if hasattr(ann.image, 'name') else ann.image
            img_path = os.path.join(self.images_dir, file_name)
            if not os.path.exists(img_path):
                print(f"Warning: File {img_path} not found. Skipping sample.")
                excluded_count += 1
                continue
            
            # Check if at least one person has valid keypoint annotations.
            has_valid_annotation = False
            if hasattr(ann, 'annorect'):
                annorect = ann.annorect
                if not isinstance(annorect, list):
                    annorect = [annorect]
                for person in annorect:
                    if hasattr(person, 'annopoints') and person.annopoints is not None:
                        kp = self.parse_annorect(person)
                        if kp is not None and np.any(kp[:, 2] > 0):
                            has_valid_annotation = True
                            break
            
            if has_valid_annotation:
                valid_annolist.append(ann)
            else:
                #print(f"Info: No valid keypoints found for {img_path}. Excluding sample.")
                excluded_count += 1
        
        self.annolist = valid_annolist
        print(f"Total images in annotation: {total}")
        print(f"Images kept: {len(valid_annolist)}")
        print(f"Images excluded: {excluded_count}")

    def __len__(self):
        return len(self.annolist)

    def __getitem__(self, idx):
        ann = self.annolist[idx]
        file_name = ann.image.name if hasattr(ann.image, 'name') else ann.image
        img_path = os.path.join(self.images_dir, file_name)
        image = Image.open(img_path).convert('RGB')
        
        original_width, original_height = image.size
        scale_x = IMAGE_SIZE[1] / original_width
        scale_y = IMAGE_SIZE[0] / original_height
        
        persons_keypoints = []
        persons_bbox = []
        
        if hasattr(ann, 'annorect'):
            annorect = ann.annorect
            if not isinstance(annorect, list):
                annorect = [annorect]
            for person in annorect:
                if hasattr(person, 'annopoints') and person.annopoints is not None:
                    kp = self.parse_annorect(person)
                    if kp is not None:
                        # Rescale keypoints.
                        kp[:, 0] *= scale_x
                        kp[:, 1] *= scale_y
                        persons_keypoints.append(kp)
                        # Attempt to get bounding box from objpos and scale.
                        if hasattr(person, 'objpos') and hasattr(person, 'scale'):
                            try:
                                if hasattr(person.objpos, 'x'):
                                    objpos = np.array([float(person.objpos.x), float(person.objpos.y)])
                                else:
                                    objpos = np.array(person.objpos)
                                scale_val = float(person.scale)
                                bb = compute_bbox(objpos, scale_val)
                                bb[0] *= scale_x
                                bb[1] *= scale_y
                                bb[2] *= scale_x
                                bb[3] *= scale_y
                                persons_bbox.append(bb)
                            except Exception as e:
                                persons_bbox.append(np.array([0, 0, IMAGE_SIZE[1], IMAGE_SIZE[0]], dtype=np.float32))
                        else:
                            persons_bbox.append(np.array([0, 0, IMAGE_SIZE[1], IMAGE_SIZE[0]], dtype=np.float32))
            if len(persons_keypoints) == 0:
                persons_keypoints.append(np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32))
                persons_bbox.append(np.array([0, 0, IMAGE_SIZE[1], IMAGE_SIZE[0]], dtype=np.float32))
        else:
            persons_keypoints.append(np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32))
            persons_bbox.append(np.array([0, 0, IMAGE_SIZE[1], IMAGE_SIZE[0]], dtype=np.float32))
        
        # Debug: visualize all persons' ground truth keypoints.
        if self.debug:
            resized_image = image.resize(IMAGE_SIZE)
            visualize_ground_truth(resized_image, persons_keypoints)
        
        # Generate aggregated heatmaps and PAFs.
        heatmaps = generate_heatmap(persons_keypoints, IMAGE_SIZE, sigma=2)
        # For PAFs, here we simply use the first person's keypoints.
        pafs = generate_paf(persons_keypoints[0], SKELETON, IMAGE_SIZE, limb_width=1)
        
        if self.transform:
            image = self.transform(image)
        
        # Convert heatmaps and PAFs to tensors.
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)
        pafs = torch.tensor(pafs, dtype=torch.float32)
        
        # Convert the first person's keypoints to a tensor for evaluation.
        gt_keypoints_tensor = torch.tensor(persons_keypoints[0], dtype=torch.float32)
        
        # Convert the first bounding box from persons_bbox into a tensor.
        bbox_tensor = torch.tensor(persons_bbox[0], dtype=torch.float32) if persons_bbox else torch.tensor([0, 0, IMAGE_SIZE[1], IMAGE_SIZE[0]], dtype=torch.float32)
        
        return image, heatmaps, pafs, gt_keypoints_tensor, bbox_tensor



    def parse_annorect(self, annorect):
        """
        Parse the annorect structure to extract keypoints.
        
        Returns:
            A numpy array of shape (NUM_KEYPOINTS, 3) with (x, y, visibility) for each keypoint.
            Assumes MPII annotations are 1-indexed and converts them to 0-indexed.
        """
        if not hasattr(annorect, 'annopoints') or annorect.annopoints is None:
            return None
        annopoints = annorect.annopoints
        # Check if annopoints has a 'point' attribute; if not, assume it's already a list/array of points.
        if hasattr(annopoints, 'point'):
            pts = annopoints.point
        else:
            pts = annopoints

        # Convert to list if needed.
        if isinstance(pts, np.ndarray):
            pts = pts.tolist()
        if not isinstance(pts, list):
            pts = [pts]

        keypoints = {}
        for pt in pts:
            try:
                # Subtract 1 to convert from 1-indexed to 0-indexed.
                kp_id = int(pt.id) - 1 if hasattr(pt, 'id') else int(pt['id']) - 1
                x = float(pt.x) if hasattr(pt, 'x') else float(pt['x'])
                y = float(pt.y) if hasattr(pt, 'y') else float(pt['y'])
            except Exception as e:
                continue
            v = 1  # Assume the keypoint is visible.
            keypoints[kp_id] = (x, y, v)

        kps = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        for i in range(NUM_KEYPOINTS):
            if i in keypoints:
                kps[i] = keypoints[i]
        return kps

# ----------------------- Model Definition with Multi-Stage Refinement -----------------------
class RefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefinementModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=1)
        self.conv6 = nn.Conv2d(128, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = self.conv6(x)
        return x


class PoseEstimationModel(nn.Module):
    def __init__(self, num_keypoints=NUM_KEYPOINTS, num_limbs=NUM_LIMBS):
        super(PoseEstimationModel, self).__init__()
        # Load pretrained ResNet-50 and remove avgpool and fc layers.
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # At this point, the feature map has 2048 channels.
        
        # Initial stage heads: map from 2048 to 256, then to desired output.
        self.initial_heatmap_head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_keypoints, kernel_size=1)
        )
        self.initial_paf_head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_limbs * 2, kernel_size=1)
        )
        # Concatenate the backbone features with initial predictions.
        in_channels_refine = 2048 + num_keypoints + num_limbs * 2
        self.refinement_heatmap = RefinementModule(in_channels_refine, num_keypoints)
        self.refinement_paf = RefinementModule(in_channels_refine, num_limbs * 2)
        
    def forward(self, x):
        features = self.backbone(x)  # Expecting feature map of shape (B, 2048, H/32, W/32)
        init_heatmaps = self.initial_heatmap_head(features)
        init_pafs = self.initial_paf_head(features)
        concat = torch.cat([features, init_heatmaps, init_pafs], dim=1)
        refined_heatmaps = self.refinement_heatmap(concat)
        refined_pafs = self.refinement_paf(concat)
        # Upsample to the original image size.
        refined_heatmaps = nn.functional.interpolate(refined_heatmaps, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
        refined_pafs = nn.functional.interpolate(refined_pafs, size=IMAGE_SIZE, mode='bilinear', align_corners=False)
        return refined_heatmaps, refined_pafs

# ----------------------- Training and Evaluation -----------------------
def train(model, dataloader, optimizer, device, epochs=10):
    # Create a StepLR scheduler: reduce LR by factor of 0.1 every 5 epochs.
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    model.train()
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, gt_heatmaps, gt_pafs, gt_keypoints, bbox) in enumerate(dataloader):
            
            images = images.to(device)
            gt_heatmaps = gt_heatmaps.to(device)
            gt_pafs = gt_pafs.to(device)
            
            optimizer.zero_grad()
            pred_heatmaps, pred_pafs = model(images)
            loss_heatmap = criterion(pred_heatmaps, gt_heatmaps)
            loss_paf = criterion(pred_pafs, gt_pafs)
            loss = loss_heatmap + loss_paf
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
        run.log({"loss":avg_loss})
        # Step the scheduler at the end of each epoch.
        scheduler.step()
        
        evaluate(model, dataloader, device)
    run.finish()

def evaluate(model, dataloader, device, max_batches=100):
    model.eval()
    oks_scores = []
    batch_count = 0
    with torch.no_grad():
        for images, gt_heatmaps, gt_pafs, gt_keypoints, bbox in dataloader:
            images = images.to(device)
            gt_keypoints = gt_keypoints.cpu().numpy()
            bbox = bbox.cpu().numpy()  # shape (batch, 4)
            pred_heatmaps, _ = model(images)
            pred_keypoints_batch = get_keypoints_from_heatmaps(pred_heatmaps)
            for gt, pred, bb in zip(gt_keypoints, pred_keypoints_batch, bbox):
                oks_val = getOKS(gt, pred, bb,SIGMAS)
                oks_scores.append(oks_val)
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Evaluation: Processed {batch_count} batches...")
            if batch_count >= max_batches:
                break
    avg_oks = np.mean(oks_scores) if oks_scores else 0
    print(f"Evaluation OKS (subset): {avg_oks:.4f}")
    model.train()

# ----------------------- Main Function -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ColorJitter(*COLOR_JITTER_PARAMS),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    mat_file = "mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"
    images_dir = "images/"
    pred_dir = "predictions/"
    
    dataset = MPIIDataset(mat_file, images_dir, transform=transform, debug=False)
    print("Dataset size:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=32)
    
    model = PoseEstimationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {pytorch_total_params}")
    train(model, dataloader, optimizer, device, epochs=10)
    
    model.eval()
    with torch.no_grad():
        for images, gt_heatmaps, gt_pafs, gt_keypoints, bbox in dataloader:
            images = images.to(device)
            pred_heatmaps, _ = model(images)
            pred_keypoints_batch = get_keypoints_from_heatmaps(pred_heatmaps)
            visualize_predictions(images[0], pred_keypoints_batch[0], pred_dir)
            break

if __name__ == "__main__":
    main()