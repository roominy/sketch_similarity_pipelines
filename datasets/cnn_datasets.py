import random
import torch
import base64
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# global transform used by all the datasets
global_transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.Grayscale(),
        transforms.Lambda(lambda img: Image.fromarray(255 - np.array(img))),  # Invert colors
        transforms.ToTensor()
    ])


# testing DeepSketch dataset
class SketchTestingDataset(Dataset):
    def __init__(self, image_paths, labels, global_mean, global_std):
        self.image_paths = image_paths
        self.labels = labels
        self.global_transform = global_transform
        self.global_mean = global_mean
        self.global_std = global_std

        self.normalize = transforms.Normalize(mean=[global_mean], std=[global_std])

    def __len__(self):
        return len(self.image_paths)

    def create_variants(self, image):
        h, w = image.shape[-2], image.shape[-1]
        canvas_size = (1, 224, 224)
        canvas = torch.full(canvas_size, 0.0)
        center_y, center_x = (canvas_size[1] - h) // 2, (canvas_size[2] - w) // 2

        variants = []
        for flip in [False, True]:
            for y in [0, canvas_size[1] - h]:
                for x in [0, canvas_size[2] - w]:
                    variant = canvas.clone()
                    if flip:
                        variant[:, y:y + h, x:x + w] = torch.flip(image, [2])
                    else:
                        variant[:, y:y + h, x:x + w] = image
                    variants.append(variant)
            center_variant = canvas.clone()
            if flip:
                center_variant[:, center_y:center_y + h, center_x:center_x + w] = torch.flip(image, [2])
            else:
                center_variant[:, center_y:center_y + h, center_x:center_x + w] = image
            variants.append(center_variant)

        return torch.stack(variants, dim=0)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        image = self.global_transform(image)

        variants = self.create_variants(image)
        variants = self.normalize(variants)

        # Normalize the variants
        # variants = [self.normalize(var) for var in variants]
        return variants, label


# training DeepSketch dataset
class SketchTrainingDataset(Dataset):

    def __init__(self, image_paths, labels, global_mean, global_std):
        self.image_paths = image_paths
        self.labels = labels
        self.global_transform = global_transform
        self.global_mean = global_mean
        self.global_std = global_std

        # Define transforms for random rotation and horizontal flip
        self.transform = transforms.Compose([
            transforms.RandomApply([transforms.RandomRotation(degrees=(-35, 35))]),
            transforms.RandomHorizontalFlip()
        ])

        # Normalize should be applied separately on tensor
        self.normalize = transforms.Normalize(mean=[global_mean], std=[global_std])

    def __len__(self):
        return len(self.image_paths)

    def create_transformed_image(self, image):
        canvas_size = (1, 224, 224)
        canvas = torch.full(canvas_size, 0.0)

        h, w = image.shape[-2], image.shape[-1]

        max_y = canvas_size[1] - h
        max_x = canvas_size[2] - w
        top = random.randint(0, max_y)
        left = random.randint(0, max_x)

        canvas[:, top:top + h, left:left + w] = image

        # Apply the random transformations
        canvas = self.transform(canvas)

        # Normalize the canvas tensor
        canvas = self.normalize(canvas)

        return canvas

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        image = self.global_transform(image)
        transformed_image = self.create_transformed_image(image)
        return transformed_image, label

# validation DeepSketch dataset
class SketchValidationDataset(Dataset):

    def __init__(self, image_paths, labels, global_mean, global_std):
        self.image_paths = image_paths
        self.labels = labels
        self.global_transform = global_transform
        self.global_mean = global_mean
        self.global_std = global_std

        # Normalize should be applied separately on tensor
        self.normalize = transforms.Normalize(mean=[global_mean], std=[global_std])

    def __len__(self):
        return len(self.image_paths)

    def create_transformed_image(self, image):
        canvas_size = (1, 224, 224)
        canvas = torch.full(canvas_size, 0.0)

        h, w = image.shape[-2], image.shape[-1]

        max_y = canvas_size[1] - h
        max_x = canvas_size[2] - w
        top = (canvas_size[1] - h) // 2
        left = (canvas_size[2] - w) // 2

        canvas[:, top:top + h, left:left + w] = image

        # Normalize the canvas tensor
        canvas = self.normalize(canvas)

        return canvas

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        image = self.global_transform(image)
        transformed_image = self.create_transformed_image(image)
        return transformed_image, label

# dataset to encode sketches
class SketchDataset(Dataset):

    def __init__(self, image_paths, labels, global_mean, global_std):
        self.image_paths = image_paths
        self.labels = labels
        self.global_transform = global_transform
        self.global_mean = global_mean
        self.global_std = global_std

        # Normalize should be applied separately on tensor
        self.normalize = transforms.Normalize(mean=[global_mean], std=[global_std])

    def __len__(self):
        return len(self.image_paths)

    def create_transformed_image(self, image):
        canvas_size = (1, 224, 224)
        canvas = torch.full(canvas_size, 0.0)

        h, w = image.shape[-2], image.shape[-1]

        center_y, center_x = (canvas_size[1] - h) // 2, (canvas_size[2] - w) // 2
        canvas[:, center_y:center_y + h, center_x:center_x + w] = image

        # Normalize the canvas tensor
        canvas = self.normalize(canvas)

        return canvas

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        image = self.global_transform(image)
        transformed_image = self.create_transformed_image(image)
        return transformed_image, label
