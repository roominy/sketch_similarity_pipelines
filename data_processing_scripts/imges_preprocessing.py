import os
from rembg import remove, new_session
from PIL import Image, ImageDraw, ImageEnhance
from tqdm import tqdm
import subprocess
import matplotlib.pyplot as plt
import settings

# processed = ['axe', 'hammer', 'couch', 'fish', 'teapot', 'harp', 'megaphone', 'bicycle', 'duck']


processed = []


def apply_threshold_to_alpha(img, threshold=128):
    # separate the alpha channel
    r, g, b, alpha = img.split()

    # apply the threshold to the alpha channel
    alpha = alpha.point(lambda p: p > threshold and 255)

    # recombine the image with the modified alpha channel
    img.putalpha(alpha)

    # get the bounding box with the new alpha
    bbox = img.getbbox()
    return bbox


def resize_image(img, out_resize):
    # Resize the image
    resized_img = img.resize(out_resize)

    return resized_img


def center_image(img, bbox, padding=2, bg=(255, 255, 255, 255), show=False):
    # calculate the width and height of the bounding box
    object_width = bbox[2] - bbox[0]
    object_height = bbox[3] - bbox[1]

    object_center_x = bbox[0] + object_width // 2
    object_center_y = bbox[1] + object_height // 2

    # determine the size of the square canvas + padding
    canvas_size = max(object_width, object_height) + (padding * 2)

    # create a new square with background , bg=(0,0,0,0) for transparent
    centered_img = Image.new("RGBA", (canvas_size, canvas_size), bg)

    # calculate the center of the image
    image_center_x = canvas_size // 2
    image_center_y = canvas_size // 2

    # calculate the new position to paste the object, centered in the square image
    paste_x = image_center_x - object_center_x
    paste_y = image_center_y - object_center_y

    # Paste the object in the center of the new square image
    centered_img.paste(img, (paste_x, paste_y), img)

    if show:
        print("Bounding box:", bbox)
        print("Object dimensions:", object_width, object_height)
        print("Canvas size:", canvas_size)
        print("Paste dimensions:", paste_x, paste_y)

    return centered_img


def add_bg_center_resize_image(file_path, padding=15, threshold=128, out_resize=(256, 256), bg=(255, 255, 255, 255),
                               show=False):
    # load the image
    img = Image.open(file_path).convert("RGBA")

    # find the bounds of the object
    bbox = apply_threshold_to_alpha(img.copy(), threshold=threshold)

    # center  the object in a new square image and add background
    centered_img = center_image(img, bbox, padding=padding, bg=bg, show=show)

    resized_img = resize_image(centered_img, out_resize)

    if show:
        # Draw the bounding box on the image
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline="red")
        # Display the image with the bounding box
        plt.imshow(img)
        plt.show()

        # Display the final image
        plt.imshow(resized_img)
        plt.show()
    return resized_img.convert('RGB')
    # sharpness_image = ImageEnhance.Sharpness(resized_img)
    # sharpness_image = sharpness_image.enhance(5)
    # return sharpness_image.convert('RGB')


def process_images_in_subfolders(path, input_folder, output_folder, out_prefix="processed"):
    in_folder_path = os.path.join(path, input_folder)
    out_folder_path = os.path.join(path, output_folder)

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
                output_file_path = os.path.join(out_subfolder_path,
                                                f"{out_prefix}-{os.path.splitext(file_name)[0].split('-')[-1]}.jpeg")

                processed_img = add_bg_center_resize_image(input_file_path, padding=15, threshold=128,
                                                           out_resize=(256, 256), bg=(255, 255, 255, 255), show=False)
                processed_img.save(output_file_path)


def bg_remove_in_subfolders(session, path, input_folder, output_folder, out_prefix="rmbg"):
    in_folder_path = os.path.join(path, input_folder)
    out_folder_path = os.path.join(path, output_folder)

    # Get all directories in the input folder
    directories = [d for d in os.listdir(in_folder_path) if os.path.isdir(os.path.join(in_folder_path, d))]

    # Iterate over all directories with a progress bar
    for folder_name in tqdm(directories, desc="Processing folders"):
        in_subfolder_path = os.path.join(in_folder_path, folder_name)
        out_subfolder_path = os.path.join(out_folder_path, folder_name)

        # Check if it's a directory
        if os.path.isdir(in_subfolder_path) and folder_name not in processed:

            if not os.path.exists(out_subfolder_path):
                os.makedirs(out_subfolder_path)

            for file_name in os.listdir(in_subfolder_path):
                input_file_path = os.path.join(in_subfolder_path, file_name)
                output_file_path = os.path.join(out_subfolder_path,
                                                f"{out_prefix}-{os.path.splitext(file_name)[0]}.png")

                if os.path.isfile(input_file_path) and os.path.splitext(file_name)[1] in settings.IMG_EXTENSIONS:
                    input_image = Image.open(input_file_path)
                    output_image = remove(input_image, session=session)
                    output_image.save(output_file_path)
                    print(f"Processed {input_file_path} to {output_file_path}")



config = {}
config['images_path'] = '/path/to/images_folders'
config['input_images_folder'] = 'original_images'
config['bg_removed_images_folder'] = 'bg_removed_images'
config['processed_images_folder'] = "processed_images"




if __name__ == "__main__":
    # Define the input and output folders
    session = new_session()
    images_path = config['images_path']
    # path = "/data/roominy/project_code/objects_dataset"
    input_folder = config['input_images_folder']

    # define the remove images background output folders
    bg_removed_folder = config['output_images_folder']


    # define the processed images output folders
    processed_folder = config['processed_images_folder']


    print('* remove images background with rembg')
    # remove images background  with rembg
    bg_remove_in_subfolders(session, images_path, input_folder, bg_removed_folder)

    print('* process the images from bg_remove ')
    # process the images from rembg
    process_images_in_subfolders(images_path, bg_removed_folder, processed_folder, out_prefix="processed")



