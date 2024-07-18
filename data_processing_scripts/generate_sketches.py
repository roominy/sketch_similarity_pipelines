import os
from tqdm import tqdm
from datasets.photo_sketching_datasets import TestDirDataset
from torch.utils.data import DataLoader
from models.pix2pix_model import Pix2PixModel
import skimage.morphology
import skimage.io
import skimage.util
import numpy as np
import PIL.Image as Image
import argparse
import settings

processed = []

def contour_in_subfolders(path, input_folder, output_folder, out_prefix="contour"):
    """
    Generate contour images using the Pix2Pix model and save them in the output folder.

    Args:
    path (str): Root path containing input and output folders.
    input_folder (str): Name of the input folder.
    output_folder (str): Name of the output folder.
    out_prefix (str): Prefix for output filenames.
    """
    in_folder_path = os.path.join(path, input_folder)
    out_folder_path = os.path.join(path, output_folder)

    # Initialize the Pix2Pix model
    model = Pix2PixModel()

    # Get all directories in the input folder
    directories = [d for d in os.listdir(in_folder_path) if os.path.isdir(os.path.join(in_folder_path, d))]

    # Iterate over all directories with a progress bar
    for folder_name in tqdm(directories, desc="Processing folders"):
        in_subfolder_path = os.path.join(in_folder_path, folder_name)
        out_subfolder_path = os.path.join(out_folder_path, folder_name)

        # Check if it's a directory and not already processed
        if os.path.isdir(in_subfolder_path) and folder_name not in processed:
            if not os.path.exists(out_subfolder_path):
                os.makedirs(out_subfolder_path)

            dataset = TestDirDataset(dataroot=in_subfolder_path)
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

            for i, data in enumerate(data_loader):
                output = model.test(data)
                model.write_image(data, output, out_subfolder_path)

def thinning_contours_in_subfolders(path, input_folder, output_folder, out_prefix="thinned"):
    """
    Apply thinning to contour images and save them in the output folder.

    Args:
    path (str): Root path containing input and output folders.
    input_folder (str): Name of the input folder.
    output_folder (str): Name of the output folder.
    out_prefix (str): Prefix for output filenames.
    """
    in_folder_path = os.path.join(path, input_folder)
    out_folder_path = os.path.join(path, output_folder)

    # Get all directories in the input folder
    directories = [d for d in os.listdir(in_folder_path) if os.path.isdir(os.path.join(in_folder_path, d))]

    # Iterate over all directories with a progress bar
    for folder_name in tqdm(directories, desc="Processing folders"):
        in_subfolder_path = os.path.join(in_folder_path, folder_name)
        out_subfolder_path = os.path.join(out_folder_path, folder_name)

        if not os.path.exists(out_subfolder_path):
            os.makedirs(out_subfolder_path)

        # Get all image files in the directory
        image_files = [f for f in os.listdir(in_subfolder_path) if os.path.splitext(f)[1].lower() in settings.IMG_EXTENSIONS]

        # Iterate over all files with a progress bar
        for file_name in tqdm(image_files, desc=f"Processing images in {folder_name}"):
            input_file_path = os.path.join(in_subfolder_path, file_name)
            output_file_path = os.path.join(out_subfolder_path, f"{out_prefix}-{os.path.splitext(file_name)[0].split('-')[-1]}.jpeg")

            try:
                img = skimage.io.imread(input_file_path)
                image = skimage.util.invert(img)
                image = image > 125  # Thresholding
                skel = skimage.morphology.skeletonize(image)  # Thinning
                thinned_image = (1 - np.array(skel).astype(np.uint8)) * 255  # Convert back to original intensity range
                thinned_image = Image.fromarray(thinned_image)
                thinned_image.save(output_file_path)
            except Exception as e:
                print(f"Error processing {input_file_path}: {str(e)}")

def parse_args():
    """
    Parse command-line arguments.

    Returns:
    Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Generate and thin contour images.')
    parser.add_argument('--images_path', type=str, required=True, help='Root path containing images folders.')
    parser.add_argument('--processed_images_folder', type=str, required=True, help='Folder containing processed images.')
    parser.add_argument('--contour_sketches_folder', type=str, required=True, help='Folder to save contour sketches.')
    parser.add_argument('--thinned_sketches_folder', type=str, required=True, help='Folder to save thinned sketches.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Extract paths from command-line arguments
    images_path = args.images_path
    processed_input_folder = args.processed_images_folder
    contour_sketches_folder = args.contour_sketches_folder
    thinned_sketches_folder = args.thinned_sketches_folder

    print('* Generate contour images with Pix2Pix model')
    # Generate contour images using Pix2Pix model
    contour_in_subfolders(images_path, processed_input_folder, contour_sketches_folder)

    print('* Thinning the contour images')
    # Apply thinning to contour images
    thinning_contours_in_subfolders(images_path, contour_sketches_folder, thinned_sketches_folder)
