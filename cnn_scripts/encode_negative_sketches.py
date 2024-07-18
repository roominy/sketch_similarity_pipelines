import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import argparse
from datasets.cnn_datasets import SketchDataset
from models.deepsketch3_model import ResNet18WithDropout
import settings

def load_image_paths_and_labels(dataset_root):
    """
    Load image paths and corresponding name labels from the dataset directory.

    Args:
    dataset_root (str): Root directory of the dataset.

    Returns:
    tuple: A tuple containing lists of image paths and name labels.
    """
    image_paths = []
    name_labels = []
    files = os.listdir(dataset_root)
    for file in files:
        file_splits = file.split("-")
        image_paths.append(os.path.join(dataset_root, file))
        name_labels.append(file_splits[0])
    return image_paths, name_labels

def parse_args():
    """
    Parse command-line arguments.

    Returns:
    Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process dataset with ResNet18.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output encodings and labels.')
    parser.add_argument('--dataset_type', type=str, default='negative_user', required=True, help='Dataset type (e.g., negative_user).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Extract arguments
    dataset_path = args.dataset_path
    model_path = args.model_path
    output_path = args.output_path
    dataset_type = args.dataset_type
    global_mean = settings.GLOBAL_MEAN
    global_std = settings.GLOBAL_STDEV

    # Load the pretrained model with the specified configuration
    model = ResNet18WithDropout(pretrained=True, output_encoding=True)
    loaded_checkpoint = torch.load(model_path)
    model.load_state_dict(loaded_checkpoint["state_dict"])
    model.eval()

    # Load image paths and labels from the dataset
    image_paths, name_labels = load_image_paths_and_labels(dataset_path)

    # Create a dataset and dataloader for processing the images
    dataset = SketchDataset(image_paths, name_labels, global_mean, global_std)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    encodings = []

    # Process each batch of images and extract features using the model
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            features, _ = model(images)
            encodings.append(features.cpu().numpy())

    # Concatenate all encodings into a single array
    encodings = np.concatenate(encodings, axis=0)

    # Save the encodings and name labels to the specified output directory
    np.save(os.path.join(output_path, f"tu_berlin_encodings_{dataset_type}.npy"), encodings)
    np.save(os.path.join(output_path, f"tu_berlin_name_labels_{dataset_type}.npy"), name_labels)
