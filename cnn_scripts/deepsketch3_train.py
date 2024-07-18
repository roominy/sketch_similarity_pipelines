import csv
from datasets.cnn_datasets import SketchTestingDataset, SketchValidationDataset, SketchTrainingDataset
import datetime as dt
import json
import matplotlib.pyplot as plt
from models.deepsketch3_model import ResNet18WithDropout
import os
import pandas as pd
import settings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train and evaluate a deep learning model for sketch recognition.')
    parser.add_argument('--n_splits', type=int, default=3, help='Number of data splits.')
    parser.add_argument('--train_ratio', type=float, default=0.67, help='Training data ratio.')
    parser.add_argument('--experiment_dir', type=str, required=True, help='Directory to save experiment results.')
    parser.add_argument('--splits_dir', type=str, required=True, help='Directory containing data splits.')
    parser.add_argument('--dropout_rate', type=float, default=0.65, help='Dropout rate for the model.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer.')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs.')
    return parser.parse_args()


def load_model(dropout_rate=0.5, load_weights=True):
    """
    Load the ResNet18 model with dropout.

    Args:
    dropout_rate (float): Dropout rate for the model.
    load_weights (bool): Whether to load pretrained weights.

    Returns:
    model: The loaded model.
    """
    model = ResNet18WithDropout(dropout_rate=dropout_rate, pretrained=load_weights)
    return model


def train_model(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
    model: The model to train.
    train_loader (DataLoader): DataLoader for training data.
    criterion: Loss function.
    optimizer: Optimizer.
    device: Device to use for training.

    Returns:
    epoch_loss (float): Training loss for the epoch.
    epoch_acc (float): Training accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_model(model, dataloader, criterion, device):
    """
    Validate the model.

    Args:
    model: The model to validate.
    dataloader (DataLoader): DataLoader for validation data.
    criterion: Loss function.
    device: Device to use for validation.

    Returns:
    loss (float): Validation loss.
    accuracy (float): Validation accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def predict_and_evaluate(model, dataloader, device):
    """
    Predict and evaluate the model.

    Args:
    model: The model to evaluate.
    dataloader (DataLoader): DataLoader for testing data.
    device: Device to use for evaluation.

    Returns:
    accuracy (float): Accuracy of the model.
    predicted_classes (list): List of predicted classes.
    """
    model.to(device)
    model.eval()

    predicted_classes = []
    ground_truth = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Testing'):
            variants, label = batch
            variants, label = variants.squeeze(0).to(device), label.to(device)
            batch_probabilities = model(variants)
            pred_probabilities = torch.mean(F.softmax(batch_probabilities, dim=1), dim=0)
            predicted_classes.append(torch.argmax(pred_probabilities).item())
            ground_truth.append(label.item())

    accuracy = sum([pred == true for pred, true in zip(predicted_classes, ground_truth)]) / len(ground_truth)

    return accuracy, predicted_classes


def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pth"):
    """
    Save the model checkpoint.

    Args:
    state (dict): State dictionary containing model parameters.
    is_best (bool): Flag indicating if this is the best model so far.
    checkpoint_path (str): Path to save the checkpoint.
    filename (str): Filename for the checkpoint.
    """
    if is_best:
        filename = "best_" + filename
    checkpoint_path = os.path.join(checkpoint_path, filename)
    torch.save(state, checkpoint_path)


def plot_metrics(train_metrics, val_metrics, metric_name, plot_path):
    """
    Plot training and validation metrics.

    Args:
    train_metrics (list): List of training metrics.
    val_metrics (list): List of validation metrics.
    metric_name (str): Name of the metric to plot.
    plot_path (str): Path to save the plot.
    """
    plt.figure()
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo-', label=f'Training {metric_name}')
    plt.plot(epochs, val_metrics, 'ro-', label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_mean = settings.GLOBAL_MEAN
    global_std = settings.GLOBAL_STDEV
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    momentum = args.momentum
    epochs = args.epochs
    datatime_str = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for split_idx in range(args.n_splits):
        experiment = split_idx + 1
        train_filename = os.path.join(args.splits_dir,
                                      f"train_set_split_{args.train_ratio}_{experiment}.csv")
        test_filename = os.path.join(args.splits_dir,
                                     f"test_set_split_{args.train_ratio}_{experiment}.csv")

        train_df = pd.read_csv(train_filename)
        test_df = pd.read_csv(test_filename)

        train_image_paths = train_df['image_paths'].to_list()
        train_labels = train_df['labels'].to_list()
        test_image_paths = test_df['image_paths'].to_list()
        test_labels = test_df['labels'].to_list()

        # Create datasets
        sketch_train_dataset = SketchTrainingDataset(image_paths=train_image_paths, labels=train_labels,
                                                     global_mean=global_mean, global_std=global_std)
        sketch_val_dataset = SketchValidationDataset(image_paths=test_image_paths, labels=test_labels,
                                                     global_mean=global_mean, global_std=global_std)
        sketch_test_dataset = SketchTestingDataset(image_paths=test_image_paths, labels=test_labels,
                                                   global_mean=global_mean, global_std=global_std)

        train_dataloader = DataLoader(sketch_train_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
        val_dataloader = DataLoader(sketch_val_dataset, batch_size=batch_size, shuffle=False, num_workers=20)
        test_dataloader = DataLoader(sketch_test_dataset, batch_size=1, shuffle=False, num_workers=20)

        model = load_model(dropout_rate=args.dropout_rate, load_weights=True).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        exp_name = (f"experiment-{experiment}-{batch_size}-{learning_rate}"
                    + f"-{str(args.dropout_rate)}_{str(args.train_ratio)}_{datatime_str}")
        experiment_dir = os.path.join(args.experiment_dir, exp_name)
        checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        config_path = os.path.join(experiment_dir, 'config.json')
        with open(config_path, 'w') as config_file:
            json.dump(vars(args), config_file, indent=4)

        best_val_loss = np.inf
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies, test_accuracies = [], [], []

        best_epoch = 0
        best_model_state_dict = None
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss, train_acc = train_model(model, train_dataloader, criterion, optimizer, device)
            val_loss, val_acc = validate_model(model, val_dataloader, criterion, device)
            test_acc, _ = predict_and_evaluate(model, test_dataloader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            test_accuracies.append(test_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_filename = f"checkpoint_epoch_{epoch + 1}.pth"
                checkpoint_state = {
                    'global_mean': global_mean,
                    'global_std': global_std,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_acc': val_acc,
                }
                save_checkpoint(checkpoint_state, True, checkpoints_dir, filename=checkpoint_filename)

            # Save checkpoint every 'settings.CHECKPOINTS_EVERY' epochs
            if (epoch + 1) % settings.CHECKPOINTS_EVERY == 0:
                checkpoint_filename = f"checkpoint_epoch_{epoch + 1}.pth"
                checkpoint_state = {
                    'global_mean': global_mean,
                    'global_std': global_std,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_acc': val_acc,
                }
                save_checkpoint(checkpoint_state, False, checkpoints_dir, filename=checkpoint_filename)

            plot_metrics(train_losses, val_losses, 'Loss', os.path.join(experiment_dir, 'loss_plot.png'))
            plot_metrics(train_accuracies, val_accuracies, 'Accuracy',
                         os.path.join(experiment_dir, 'accuracy_plot.png'))

        # Save metrics to CSV
        experiment_loss_path = os.path.join(experiment_dir, f'{exp_name}_loss_accuracy.csv')
        with open(experiment_loss_path, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'test_accuracy'])
            for epoch in range(len(train_losses)):
                csv_writer.writerow([epoch + 1, train_losses[epoch], val_losses[epoch], train_accuracies[epoch],
                                     val_accuracies[epoch], test_accuracies[epoch]])

        print("Training complete.")
