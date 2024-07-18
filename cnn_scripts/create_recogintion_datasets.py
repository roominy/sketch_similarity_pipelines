import os
import argparse
import pandas as pd
from torchvision import datasets
from sklearn.model_selection import train_test_split

def parse_args():
    """
    Parse command-line arguments.

    Returns:
    Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Create train/test splits for the TU Berlin dataset.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--train_ratios', type=float, nargs='+', required=True, help='List of train ratios.')
    parser.add_argument('--n_splits', type=int, default=3, help='Number of random splits.')
    parser.add_argument('--random_seeds', type=int, nargs='+', default=[58925, 27334, 87860], help='List of random seeds for each split.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for CSVs.')

    return parser.parse_args()

def main():
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the dataset from the specified directory
    dataset = datasets.ImageFolder(root=args.dataset_path)

    for train_ratio in args.train_ratios:
        # Repeat the splitting process using predefined random seeds
        for split_number, seed in enumerate(args.random_seeds):
            train_indices = []
            test_indices = []

            # Create a list of empty lists to store indices of images for each class
            class_indices = [[] for _ in range(len(dataset.classes))]

            # Populate the list with indices of images belonging to each class
            for idx, (_, label) in enumerate(dataset.imgs):
                class_indices[label].append(idx)

            # Split indices for each class into train and test sets using the specific seed
            for class_list in class_indices:
                train_idx, test_idx = train_test_split(
                    class_list,
                    test_size=1 - train_ratio,
                    random_state=seed)
                train_indices.extend(train_idx)
                test_indices.extend(test_idx)

            # Extract image paths and labels using the indices
            train_image_paths = [dataset.imgs[i][0] for i in train_indices]
            train_labels = [dataset.imgs[i][1] for i in train_indices]
            test_image_paths = [dataset.imgs[i][0] for i in test_indices]
            test_labels = [dataset.imgs[i][1] for i in test_indices]

            # Save image paths and labels to CSV
            train_df = pd.DataFrame({
                'image_paths': train_image_paths,
                'labels': train_labels
            })
            test_df = pd.DataFrame({
                'image_paths': test_image_paths,
                'labels': test_labels
            })

            # Define filenames for the CSVs
            train_filename = os.path.join(args.output_dir,
                                          f'train_set_split_{train_ratio}_{split_number + 1}.csv')
            test_filename = os.path.join(args.output_dir,
                                         f'test_set_split_{train_ratio}_{split_number + 1}.csv')

            # Save to CSV
            train_df.to_csv(train_filename, index=False)
            test_df.to_csv(test_filename, index=False)

            print(f'Split {split_number + 1}: Train and test data saved to {train_filename} and '
                  f'{test_filename} with random seed {seed}.')

if __name__ == '__main__':
    main()
