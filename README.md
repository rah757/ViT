Vision Transformer (ViT) on CIFAR-10

This repository provides an example implementation of a Vision Transformer (ViT) model trained on the CIFAR-10 dataset. It uses PyTorch for model building, training, and evaluation.

Repository Structure

project/
    README.md
    model.py
    data.py
    utils.py
    train.py
    main.py
    experiments/
        <experiment_name>/
            config.json
            metrics.json
            model_final.pt
model.py: Contains the Vision Transformer implementation including patch embeddings, the Transformer encoder, and the classification head.
data.py: Handles data loading, transformation, and preparation of the CIFAR-10 dataset.
utils.py: Utilities for saving/loading experiments, model checkpoints, visualizing images, and attention maps.
train.py: Defines a Trainer class used for training and evaluating the model.
main.py: The entry point for running training. It sets up configurations, starts training, and generates plots and visualizations.
Requirements

Python 3.7 or higher
PyTorch 1.7 or higher (with CUDA if a GPU is available)
torchvision
matplotlib
numpy
You can install the requirements via:

pip install torch torchvision matplotlib numpy
Training the Model

Prepare the dataset: The CIFAR-10 dataset is automatically downloaded by torchvision.datasets.CIFAR10 upon the first run.
Run the training:
python main.py
This will:

Load the CIFAR-10 dataset.
Initialize the Vision Transformer model.
Train the model for the specified number of epochs.
Save training metrics and checkpoints in the experiments/<experiment_name>/ directory.
Configuration

Edit the config dictionary in main.py to adjust model architecture and training hyperparameters:

patch_size: Size of image patches.
hidden_size: Dimensionality of the model’s embeddings.
num_hidden_layers: Number of Transformer blocks.
num_attention_heads: Number of attention heads per block.
intermediate_size: Dimensionality of the MLP hidden layer.
dropout probabilities: Adjust for regularization.
learning rate, batch size, epochs, etc.
Make sure the constraints (e.g., hidden_size % num_attention_heads == 0) are maintained.

Experiment Results

Training metrics (train/test loss, accuracy) and model checkpoints are saved under experiments/<experiment_name>/.

config.json: Contains the configuration used for the experiment.
metrics.json: Contains the training/testing losses and accuracies recorded for each epoch.
model_final.pt: The final trained model weights.
You can use utils.load_experiment() to load a trained model and its metrics for further analysis or inference.

Visualization

Visualizing Dataset: After training, the code (in main.py) will display random samples from the CIFAR-10 training set.
Plot Training Curves: Training and testing losses, along with accuracies, are plotted and saved to metrics.png.
Attention Maps: The script will also visualize attention maps overlayed on the input images, saved as attention.png.
Troubleshooting

If training is slow, ensure you have a GPU available and that PyTorch is using the GPU (device = "cuda").
If you run into memory issues, try lowering the batch_size or the model’s hidden dimensions.