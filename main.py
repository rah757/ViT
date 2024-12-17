import torch
from torch import nn, optim
from data import prepare_data
from model import ViTForClassfication
from train import Trainer
from utils import load_experiment, visualize_images, visualize_attention
import matplotlib.pyplot as plt

if __name__ == '__main__':
    exp_name = 'vit-with-10-epochs'
    batch_size = 32
    epochs = 10
    lr = 1e-2
    save_model_every = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "patch_size": 4,
        "hidden_size": 48,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 4*48,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "image_size": 32,
        "num_classes": 10,
        "num_channels": 3,
        "qkv_bias": True,
        "use_faster_attention": True
    }

    # Ensure configuration is correct
    assert config["hidden_size"] % config["num_attention_heads"] == 0
    assert config["intermediate_size"] == 4 * config["hidden_size"]
    assert config["image_size"] % config["patch_size"] == 0

    trainloader, testloader, classes = prepare_data(batch_size=batch_size)
    model = ViTForClassfication(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, loss_fn, exp_name, device)
    trainer.train(trainloader, testloader, epochs, config, save_model_every_n_epochs=save_model_every)

    # Visualize some data
    visualize_images()

    # Load experiment results and plot
    config_loaded, model_loaded, train_losses, test_losses, accuracies = load_experiment(exp_name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ax1.plot(train_losses, label="Train loss")
    ax1.plot(test_losses, label="Test loss")
    ax1.legend()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(accuracies, label="Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    plt.savefig("metrics.png")
    plt.show()

    # Visualize attention
    visualize_attention(model_loaded, "attention.png", device=device)
