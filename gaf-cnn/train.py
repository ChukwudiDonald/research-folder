"""
This script implements the methodology from the paper:
"ENCODING CANDLESTICKS AS IMAGES FOR PATTERN CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORKS" by Jun-Hao Chen & Yun-Cheng Tsai .

It performs the following steps:
1.  Loads financial time-series data (OHLC) from CSV files.
2.  Calculates additional candlestick features (body size, upper/lower wicks).
3.  Transforms each time-series channel into a 2D image using Gramian Angular Fields (GAF),
    preserving temporal dependencies. The GAF is defined as:
    $GAF = \\cos(\\arccos(\\tilde{x_i}) + \\arccos(\\tilde{x_j}))$
4.  Builds a PyTorch TensorDataset where each sample is a multi-channel image.
5.  Defines a Convolutional Neural Network (CNN) to classify these images.
6.  Trains the CNN on a training set and evaluates it on a validation set.
7.  Performs a final visual validation by plotting candlestick charts of new patterns
    and displaying the model's prediction.
"""

# ## 1. Imports
import torch
import torch.nn.functional as F

from torch import nn
import pandas as pd
from pathlib import Path
from pyts.image import GramianAngularField
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.pyplot as plt
import math

# ## 2. Configuration
# Path to the root directory containing class-named subdirectories of CSV files.
DATA_ROOT = "./synthetic_market_data"

# Parameters for data loading and model training
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.0001
DROPOUT_PROB = 0.15
# Define which data channels to convert into images.
# You can use the default ["Open", "High", "Low", "Close"] or the custom ones.
DATA_COLUMNS = ["Close", "Upper", "Real", "Lower"]
# DATA_COLUMNS = ["Open", "High", "Low", "Close"]


# --------------------------------------------------------------------------------

# ## 3. Data Preprocessing Functions

def csv_to_gaf_tensor(data_path, class_name, columns, gaf_transformer):
    """
    Reads a single CSV file, processes it, and converts specified time-series
    columns into a multi-channel GAF image tensor.

    Args:
        data_path (Path or str): Path to the input CSV file.
        class_name (int): The integer label for the class.
        columns (list[str]): A list of column names to be converted to image channels.
        gaf_transformer (GramianAngularField): An instance of the GAF transformer.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - x (torch.Tensor): The resulting multi-channel image tensor.
            - y (torch.Tensor): The corresponding label tensor.
    """
    df = pd.read_csv(data_path)

    # Engineer features from OHLC data to represent candlestick components
    # 'Real': The size of the candlestick body.
    df['Real'] = (df['Close'] - df['Open']).abs()
    # 'Upper': The size of the upper wick (shadow).
    df['Upper'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    # 'Lower': The size of the lower wick (shadow).
    df['Lower'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    images = []
    for col in columns:
        # Reshape the time series for the transformer
        time_series = df[col].values.reshape(1, -1)
        # Apply the GAF transformation and convert to a tensor
        img = torch.tensor(gaf_transformer.transform(time_series)[0], dtype=torch.float)
        images.append(img)

    # Stack the individual channel images into a single multi-channel tensor
    x = torch.stack(images)
    y = torch.tensor(class_name, dtype=torch.long)

    return x, y


def create_dataset(root_dir, columns):
    """
    Scans a directory for class-subfolders, processes all CSV files within them,
    and returns stacked tensors for the entire dataset.

    Args:
        root_dir (str): The path to the main data directory.
        columns (list[str]): A list of column names to process.

    Returns:
        tuple[torch.Tensor, torch.Tensor, list[str]]: A tuple containing:
            - X (torch.Tensor): A tensor of all image samples.
            - Y (torch.Tensor): A tensor of all corresponding labels.
            - class_names (list[str]): A list of the discovered class names.
    """
    X = []
    Y = []
    # Initialize the GAF transformer. This will be reused for all files.
    gaf_transformer = GramianAngularField()
    root_path = Path(root_dir)

    # Discover class names by listing subdirectories
    class_names = sorted([d.name for d in root_path.iterdir() if d.is_dir()])
    class_dict = {name: i for i, name in enumerate(class_names)}

    all_csv_files = list(root_path.rglob("*.csv"))
    print(f"Found {len(all_csv_files)} CSV files to process...")

    for file_path in all_csv_files:
        # The parent directory's name corresponds to the class
        class_name = file_path.parent.name
        if class_name in class_dict:
            label = class_dict[class_name]
            # Convert the CSV to a GAF tensor sample
            x_sample, y_sample = csv_to_gaf_tensor(
                data_path=file_path,
                class_name=label,
                columns=columns,
                gaf_transformer=gaf_transformer
            )
            X.append(x_sample)
            Y.append(y_sample)

    print("Data processing complete. ✅")
    # Stack all individual samples into two large tensors
    return torch.stack(X), torch.stack(Y), class_names


# --------------------------------------------------------------------------------

# ## 4. PyTorch Model Definition
class CNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.
    The architecture is designed to handle the multi-channel GAF images.
    """
    def __init__(self, in_channels, num_classes, dropout_prob=0.15):
        super().__init__()
        
        # Convolutional layers progressively reduce spatial dimensions and increase channel depth.
        self.conv_layers = nn.Sequential(
            # Conv 1: High resolution, few channels
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob),
            nn.MaxPool2d(2),

            # Conv 2: Medium resolution, more channels
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob),
            nn.MaxPool2d(2),
            
            # Conv 3: Low resolution, most channels
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_prob),
        )

        self.ffn = nn.Sequential(
                nn.Linear(3200, 320),
                nn.ReLU(),
                nn.Linear(320,num_classes)
            )
        

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width].
        
        Returns:
            torch.Tensor: The output logits of shape [batch, num_classes].
        """
        # Pass input through convolutional layers
        x = self.conv_layers(x)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Pass through the fully connected network
        x = self.ffn(x)
        return x

# --------------------------------------------------------------------------------

# ## 5. Data Loading and Preparation
# Create the full dataset from the source files
x_data, y_data, class_labels = create_dataset(DATA_ROOT, columns=DATA_COLUMNS)
dataset = TensorDataset(x_data, y_data)
print(f"Dataset created with shapes: X={x_data.shape}, Y={y_data.shape}")

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Data split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

# Create DataLoaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# --------------------------------------------------------------------------------

# ## 6. Model Training
# Initialize the model, loss function, and optimizer
model = CNN(in_channels=x_data.shape[1], num_classes=len(class_labels), dropout_prob=DROPOUT_PROB)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print("\nStarting model training... 🚀")
for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()
    epoch_train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for x_batch, y_batch in train_loader:
        # Forward pass
        out = model(x_batch)
        loss = criterion(out, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item() * x_batch.size(0)
        
        # Calculate training accuracy
        _, predicted = torch.max(out.data, 1)
        train_total += y_batch.size(0)
        train_correct += (predicted == y_batch).sum().item()
    
    avg_train_loss = epoch_train_loss / len(train_loader.dataset)
    train_accuracy = 100 * train_correct / train_total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    
    # --- Validation Phase ---
    model.eval()
    epoch_val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            out = model(x_batch)
            loss = criterion(out, y_batch)
            epoch_val_loss += loss.item() * x_batch.size(0)
            
            # Calculate validation accuracy
            _, predicted = torch.max(out.data, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()
            
    avg_val_loss = epoch_val_loss / len(val_loader.dataset)
    val_accuracy = 100 * val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

print("\nTraining finished. 🎉")

# --------------------------------------------------------------------------------

# ## 7. Model Evaluation and Visualization
# List of specific CSV files to use for visual validation
validation_files = [ 
    r"real_market_data/pattern_1.csv", r"real_market_data/pattern_2.csv",
    r"real_market_data/pattern_3.csv", r"real_market_data/pattern_4.csv",
    r"real_market_data/pattern_5.csv", r"real_market_data/pattern_6.csv"
]

# Process the validation files into GAF image tensors
gaf_transformer_vis = GramianAngularField()
X_vis = []
for file in validation_files:
    # We only need the image 'x', so we get the first element of the returned tuple.
    # A dummy class_name of 0 is passed as it won't be used.
    x_sample = csv_to_gaf_tensor(file, 0, gaf_transformer=gaf_transformer_vis, columns=DATA_COLUMNS)[0]
    X_vis.append(x_sample)

# Get model predictions for the validation images
model.eval()
with torch.no_grad():
    out = model(torch.stack(X_vis))
    # Convert logits to probabilities and then to predicted class indices
    probabilities = F.softmax(out, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)

# --- Plotting the Results ---
n_files = len(validation_files)
n_cols = 2
n_rows = math.ceil(n_files / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
fig.suptitle('Visual Validation: Candlestick Patterns with Model Predictions', fontsize=16, y=1.02)
axes = axes.flatten()

for i, (ax, pred_idx) in enumerate(zip(axes, predictions)):
    # Read the original OHLC data for plotting
    df = pd.read_csv(validation_files[i])
    quotes = [(idx, row['Open'], row['High'], row['Low'], row['Close']) for idx, row in df.iterrows()]
    
    # Plot the candlestick chart
    candlestick_ohlc(ax, quotes, width=0.6, colorup='g', colordown='r')
    
    # Set the title to the predicted class label
    predicted_label = class_labels[pred_idx.item()]
    ax.set_title(f"File: {Path(validation_files[i]).name}\nPrediction: {predicted_label}")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_label_text("Price")
    ax.grid(True)

# Remove any unused subplots
for i in range(n_files, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()