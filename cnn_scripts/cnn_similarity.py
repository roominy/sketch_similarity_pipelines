import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse

def get_categories_encodings(labels_path, encodings_path):
    """
    Load encodings and labels from files and organize them by category.

    Args:
    labels_path (str): Path to the labels file.
    encodings_path (str): Path to the encodings file.

    Returns:
    dict: A dictionary where keys are category labels and values are lists of encodings.
    """
    encodings = np.load(encodings_path)
    name_labels = np.load(labels_path)
    unique_name_labels = np.unique(name_labels)
    encodings_by_label = {label: [] for label in unique_name_labels}
    for encoding, name_label in zip(encodings, name_labels):
        encodings_by_label[name_label].append(encoding)
    encodings_by_label = {label: np.array(encodings) for label, encodings in encodings_by_label.items()}
    return encodings_by_label

def parse_args():
    """
    Parse command-line arguments.

    Returns:
    Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process and analyze sketch dataset encodings.')
    parser.add_argument('--dataset_path_encodings_output', type=str, required=True, help='Path to encodings output.')
    parser.add_argument('--baselines_dataset_type', type=str, required=True, help='Type of baselines dataset (e.g., baselines_original, baselines_thinned, baselines_canny).')
    parser.add_argument('--experiment_dir', type=str, required=True, help='Directory to save experiment results.')
    parser.add_argument('--threshold', type=float, default=0.7, help='Threshold for cosine similarity.')
    parser.add_argument('--user_dataset_type', type=str, required=True, help='User dataset type.')
    parser.add_argument('--negative_user_dataset_type', type=str, required=True, help='Negative user dataset type.')
    parser.add_argument('--plot_results', action='store_true', help='Flag to plot results.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Paths to load baseline and user sketch encodings and labels
    baselines_path_name_labels = os.path.join(args.dataset_path_encodings_output,
                                              f"tu_berlin_name_labels_{args.baselines_dataset_type}.npy")
    baselines_path_encodings = os.path.join(args.dataset_path_encodings_output,
                                            f"tu_berlin_encodings_{args.baselines_dataset_type}.npy")
    dataset_path_name_labels = os.path.join(args.dataset_path_encodings_output,
                                            f"tu_berlin_name_labels_{args.user_dataset_type}.npy")
    dataset_path_encodings = os.path.join(args.dataset_path_encodings_output,
                                          f"tu_berlin_encodings_{args.user_dataset_type}.npy")
    negatives_path_name_labels = os.path.join(args.dataset_path_encodings_output,
                                              f"tu_berlin_name_labels_{args.negative_user_dataset_type}.npy")
    negatives_path_encodings = os.path.join(args.dataset_path_encodings_output,
                                            f"tu_berlin_encodings_{args.negative_user_dataset_type}.npy")

    # Load encodings and organize by category
    baselines_categories_encodings = get_categories_encodings(baselines_path_name_labels, baselines_path_encodings)
    user_sketches_categories_encodings = get_categories_encodings(dataset_path_name_labels, dataset_path_encodings)
    negatives_name_labels = np.load(negatives_path_name_labels)
    negatives_encodings = np.load(negatives_path_encodings)

    # Create experiment directory
    experiment = f"final_{args.baselines_dataset_type}_{args.threshold}"
    datetime_str = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_name = f"experiment_{experiment}_{datetime_str}"
    exp_path = os.path.join(args.experiment_dir, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    # Initialize cosine similarity
    cos = nn.CosineSimilarity(dim=0, eps=1e-8)
    df_precision = pd.DataFrame(columns=['label', 'precision', 'TP', 'FP'])

    # Process each category of user sketches
    for label, user_sketches_encodings in tqdm(user_sketches_categories_encodings.items()):
        cosine_similarities = []
        l1_distances = []
        l2_distances = []
        labels = []
        sketch_types = []
        baseline_labels = []

        category_exp_path = os.path.join(exp_path, label)
        os.makedirs(category_exp_path, exist_ok=True)

        # Compare user sketches with baseline sketches
        for user_sketch_encoding in user_sketches_encodings:
            user_sketch_encoding = torch.tensor(user_sketch_encoding)
            baseline_cosine_similarities = []
            baseline_l1_distances = []
            baseline_l2_distances = []

            for baseline_encoding in baselines_categories_encodings[label]:
                baseline_encoding = torch.tensor(baseline_encoding)
                baseline_cosine_similarities.append(cos(baseline_encoding, user_sketch_encoding).item())
                baseline_l1_distances.append(torch.abs(baseline_encoding - user_sketch_encoding).sum().item())
                baseline_l2_distances.append(torch.sqrt(torch.sum((baseline_encoding - user_sketch_encoding) ** 2)).item())

            cosine_similarities.append(max(baseline_cosine_similarities))
            l1_distances.append(min(baseline_l1_distances))
            l2_distances.append(min(baseline_l2_distances))
            labels.append(label)
            baseline_labels.append(label)
            sketch_types.append("positive")

        # Compare negative user sketches with baseline sketches
        for negative_label, negative_user_sketches_encoding in zip(negatives_name_labels, negatives_encodings):
            negative_user_sketches_encoding = torch.tensor(negative_user_sketches_encoding)
            baseline_cosine_similarities = []
            baseline_l1_distances = []
            baseline_l2_distances = []

            for baseline_encoding in baselines_categories_encodings[label]:
                baseline_encoding = torch.tensor(baseline_encoding)
                baseline_cosine_similarities.append(cos(baseline_encoding, negative_user_sketches_encoding).item())
                baseline_l1_distances.append(torch.abs(baseline_encoding - negative_user_sketches_encoding).sum().item())
                baseline_l2_distances.append(torch.sqrt(torch.sum((baseline_encoding - negative_user_sketches_encoding) ** 2)).item())

            cosine_similarities.append(max(baseline_cosine_similarities))
            l1_distances.append(min(baseline_l1_distances))
            l2_distances.append(min(baseline_l2_distances))
            labels.append(negative_label)
            baseline_labels.append(label)
            sketch_types.append("negative")

        # Create dataframe with results
        df = pd.DataFrame({
            'label': labels,
            'baseline_label': baseline_labels,
            'sketch_type': sketch_types,
            'cosine_similarity': cosine_similarities,
            'l1_distance': l1_distances,
            'l2_distance': l2_distances
        })

        # Save results to CSV
        csv_path = os.path.join(category_exp_path, f"{label}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        # Calculate precision
        df['prediction'] = df['cosine_similarity'] > args.threshold
        TP = len(df[(df['prediction'] == True) & (df['sketch_type'] == 'positive')])
        FP = len(df[(df['prediction'] == True) & (df['sketch_type'] == 'negative')])
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        df_precision = pd.concat([df_precision,
                                  pd.DataFrame([{'label': label, 'precision': precision, 'TP': TP, 'FP': FP}])],
                                 ignore_index=True)

        # Save precision results to CSV
        precision_csv_path = os.path.join(exp_path, f"precision_results.csv")
        df_precision.to_csv(precision_csv_path, index=False)

        # Plot results if required
        if args.plot_results:
            metrics = ['cosine_similarity', 'l1_distance', 'l2_distance']
            metrics_label = ['Cosine Similarity', 'Manhattan Distance', 'Euclidean Distance']
            metric_binwidth = {
                'cosine_similarity': 0.01,
                'l1_distance': 5,
                'l2_distance': 0.1
            }

            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            for i, metric in enumerate(metrics):
                positive_values = df[df['sketch_type'] == 'positive'][metric]
                negative_values = df[df['sketch_type'] == 'negative'][metric]

                ax = axs[i]
                binwidth = metric_binwidth[metric]
                p_bins = np.arange(min(positive_values), max(positive_values) + binwidth, binwidth)
                n_bins = np.arange(min(negative_values), max(negative_values) + binwidth, binwidth)

                ax.hist(positive_values, bins=p_bins, alpha=0.5, label='Positive', color='b')
                ax.hist(negative_values, bins=n_bins, alpha=0.5, label='Negative', color='r')
                ax.set_xlabel(metrics_label[i])
                ax.set_ylabel('Frequency')
                ax.set_title(f'Histogram of {metrics_label[i]}')
                ax.legend(loc='upper right')
                ax.grid(True)

            plt.tight_layout()
            histogram_path = os.path.join(exp_path, f"{label}-metrics_histograms.png")
            plt.savefig(histogram_path)
            plt.show()
            print(f"Histogram saved to {histogram_path}")
