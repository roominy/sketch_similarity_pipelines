import cv2 as cv
import numpy as np
from tqdm import tqdm
import os
import argparse
import settings

processed = []

def canny_edges(path, ksize=5, sigma=0, min_threshold=50, max_threshold=200):
    """
    Apply Canny edge detection to an image.

    Args:
    path (str): Path to the input image.
    ksize (int): Kernel size for Gaussian blur.
    sigma (float): Standard deviation for Gaussian blur.
    min_threshold (int): Minimum threshold for Canny edge detection.
    max_threshold (int): Maximum threshold for Canny edge detection.

    Returns:
    numpy.ndarray: Inverted edge map of the input image.
    """
    image = cv.imread(path)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blured = cv.GaussianBlur(image_gray, (ksize, ksize), sigma)

    sigma1 = 0.8
    median = np.median(blured)
    lower = int(max(0, (1.0 - sigma1) * median))
    upper = int(min(255, (1.0 + sigma1) * median))

    edges = cv.Canny(blured, lower, upper)
    inverted_edges = 255 - edges

    return inverted_edges

def apply_canny_on_images_in_subfolders(images_path, in_folder_folder, out_folder_folder, sigma=1.5,
                                        thresholds=(50, 200), out_prefix="canny"):
    """
    Apply Canny edge detection to all images in subfolders.

    Args:
    images_path (str): Root path containing input and output folders.
    in_folder_folder (str): Name of the input folder.
    out_folder_folder (str): Name of the output folder.
    sigma (float): Standard deviation for Gaussian blur.
    thresholds (tuple): Thresholds for Canny edge detection.
    out_prefix (str): Prefix for output filenames.
    """
    in_folder_path = os.path.join(images_path, in_folder_folder)
    out_folder_path = os.path.join(images_path, out_folder_folder)

    # Get all directories in the input folder
    directories = [d for d in os.listdir(in_folder_path) if os.path.isdir(os.path.join(in_folder_path, d))]

    # Iterate over all directories with a progress bar
    for folder_name in tqdm(directories, desc="Processing folders"):
        in_subfolder_path = os.path.join(in_folder_path, folder_name)
        out_subfolder_path = os.path.join(out_folder_path, folder_name)

        if os.path.isdir(in_subfolder_path) and folder_name not in processed:
            if not os.path.exists(out_subfolder_path):
                os.makedirs(out_subfolder_path)

            # Get all image files in the directory
            image_files = [f for f in os.listdir(in_subfolder_path) if os.path.splitext(f)[1].lower() in settings.IMG_EXTENSIONS]

            # Iterate over all files with a progress bar
            for file_name in tqdm(image_files, desc=f"Processing images in {folder_name}"):
                input_file_path = os.path.join(in_subfolder_path, file_name)
                output_file_path = os.path.join(out_subfolder_path, f"{out_prefix}-{os.path.splitext(file_name)[0].split('-')[-1]}.jpeg")

                canny_img = canny_edges(input_file_path, ksize=5, sigma=sigma, min_threshold=thresholds[0], max_threshold=thresholds[1])
                cv.imwrite(output_file_path, canny_img)

def parse_args():
    """
    Parse command-line arguments.

    Returns:
    Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Apply Canny edge detection to images in subfolders.')
    parser.add_argument('--images_path', type=str, required=True, help='Root path containing images folders.')
    parser.add_argument('--processed_images_folder', type=str, required=True, help='Folder containing processed images.')
    parser.add_argument('--canny_edge_maps_folder', type=str, required=True, help='Folder to save Canny edge maps.')
    parser.add_argument('--sigma', type=float, default=1.5, help='Standard deviation for Gaussian blur.')
    parser.add_argument('--thresholds', type=int, nargs=2, default=(50, 200), help='Thresholds for Canny edge detection.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    images_path = args.images_path
    processed_images_folder = args.processed_images_folder
    canny_edge_maps_folder = args.canny_edge_maps_folder
    sigma = args.sigma
    thresholds = args.thresholds

    apply_canny_on_images_in_subfolders(images_path, processed_images_folder, canny_edge_maps_folder, sigma=sigma,
                                        thresholds=thresholds, out_prefix="canny")
