from PIL import Image
import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset
from utils.ssim import tensorify
import settings

class BaselineSketchDataset(Dataset):
    def __init__(self, data_path, category, images_dirs_paths=[], size=256 , inverted=False):
        self.size = size
        self.category = category
        self.data_path = data_path
        self.inverted = inverted
        self.images_dirs_paths = images_dirs_paths
        self.baselines_files = self._load_files()
        self.images = [self.load_image(image_files) for image_files in self.baselines_files]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image.squeeze(0), [os.path.basename(baselines_files) for baselines_files in self.baselines_files[idx]]

    def load_image(self, paths):
        """Loads and processes images from given file paths."""
        images = []
        for path in paths:
            with Image.open(path) as img:
                img = img.convert('RGB').resize((self.size, self.size))
                img_array = np.array(img)
                if self.inverted:
                    img_array = 255 - img_array
                images.append(tensorify(img_array))
        return np.stack(images, axis=1)  # Stack images along a new axis

    def _load_files(self):
        """Generates a list of image file paths for each version based on a standard naming convention."""
        baseline_files = []
        files = []
        baselines_directory = os.path.join(self.data_path, self.images_dirs_paths[0][1], self.category)
        for extension in settings.IMG_EXTENSIONS:
            files.extend([os.path.basename(image_path).split('-')[-1]
                     for image_path in glob(os.path.join(baselines_directory, '*' + extension))])
        unique_files = set(files)  # Remove duplicates
        for file_name in unique_files:
            baseline_versions = []
            for prefix, version_path in self.images_dirs_paths:
                full_path = os.path.join(self.data_path, version_path, self.category, f'{prefix}-{file_name}')
                if os.path.exists(full_path):  # Check if the file actually exists
                    baseline_versions.append(full_path)
            if baseline_versions:
                baseline_files.append(baseline_versions)
        return baseline_files

class SketchDataset(Dataset):
    def __init__(self, directory, size=256 , inverted=False):
        self.size = size
        self.inverted = inverted
        self.image_files = self._load_files(directory)
        self.images = [self.load_image(image_file, self.size, self.inverted) for image_file in self.image_files]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return tensorify(image), os.path.basename(self.image_files[idx])

    @staticmethod
    def load_image(path, size, inverted):
        with Image.open(path) as img:
            if inverted:
                return 255 - np.array(img.convert('RGB').resize((size, size)))
            else:
                return np.array(img.convert('RGB').resize((size, size)))

    @staticmethod
    def _load_files(directory):
        files = []
        for extension in IMG_EXTENSIONS:
            files.extend(glob(os.path.join(directory, '*' + extension)))
        return files