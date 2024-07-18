import torch
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
    Load image paths and corresponding labels from the dataset directory.

    Args:
    dataset_root (str): Root directory of the dataset.

    Returns:
    tuple: A tuple containing lists of image paths, labels, and name labels.
    """
    image_paths = []
    labels = []
    name_labels = []
    classes = os.listdir(dataset_root)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    for cls_name in classes:
        cls_dir = os.path.join(dataset_root, cls_name)
        for img_name in os.listdir(cls_dir):
            image_paths.append(os.path.join(cls_dir, img_name))
            labels.append(class_to_idx[cls_name])
            name_labels.append(cls_name)
    return image_paths, labels, name_labels


def parse_args():
    """
    Parse command-line arguments.

    Returns:
    Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process TU-Berlin dataset with ResNet18.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output encodings and labels.')
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type (e.g., for baselines:baselines_original, baselines_thinned, baselines_canny - or for sketches:user).')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_path = args.dataset_path
    model_path = args.model_path
    global_mean = settings.GLOBAL_MEAN
    global_std = settings.GLOBAL_STDEV

    # Load the pretrained model with the specified configuration
    model = ResNet18WithDropout(pretrained=True, output_encoding=True)
    loaded_checkpoint = torch.load(model_path)
    model.load_state_dict(loaded_checkpoint["state_dict"])
    model.eval()

    # Load image paths and labels from the dataset
    image_paths, labels, name_labels = load_image_paths_and_labels(dataset_path)

    # Create a dataset and dataloader for processing the images
    dataset = SketchDataset(image_paths, labels, global_mean, global_std)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    encodings = []
    labels = []

    # Process each batch of images and extract features using the model
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            features, _ = model(images)
            encodings.append(features.cpu().numpy())
            labels.append(targets.cpu().numpy())

    # Concatenate all encodings and labels into single arrays
    encodings = np.concatenate(encodings, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Save the encodings and labels to the specified output directory
    np.save(os.path.join(args.output_path, f"tu_berlin_encodings_{args.dataset_type}.npy"), encodings)
    np.save(os.path.join(args.output_path, f"tu_berlin_name_labels_{args.dataset_type}.npy"), name_labels)
