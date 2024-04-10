# Generative Adversarial Networks-for-Biomaterial-Discovery


**GANs for Image Generation using Biomaterial Topography Dataset**


This project contains the code to train a suite of GAN Variants for Biomaterial Discovery. It includes a PyTorch implementation of the U-Net model, several building blocks used in the model architecture, and scripts for training and logging.



### Project Structure

- `GANs - Generated Topographic Images/`: In this file, GANs-generated Images can be found.
- `GANs - Generated Topographic Images/`:

- `models/`:
    - `acgan.py`: Implementation of Auxiliary Classifier GAN (ACGAN) model with convo layers.
    - `acvanilla.py`: Implementation of the vanilla ACGAN model.
    - `discriminators.py`: Implementation of various discriminator networks used in the GAN architecture.
    - `generators.py`: Implementation of various generator networks used in the GAN architecture.
    - `models.py`: Implementation of Base Model (parent) definitions and configurations for the GAN architecture.
    - `networks.py`: Implementation of Base Network (parent) definitions and configurations for the GAN architecture.
    - `vanillagan.py`: Implementation of vanilla GAN model.
    - `wgan.py`: Implementation of Wasserstein GAN (WGAN) and WGAN-GP model.

- `options/`:
    - `base_options.py`: Basic Command-line arguments for the training script.
    - `train_options.py`: Hyperparameter Command-line arguments for the training script.

- `utils/`:
    - `images_utils.py`: Utilities for image handling.
    - `custom_layers.py`: This file contains scripts for custom layers used in the GAN architecture.
    - `losses.py`: This file contains scripts for loss functions used in the GAN training process.
    - `tb_visualizer.py`: This file provides scripts for a TensorBoard visualizer for monitoring GAN training progress.
    - `utils.py`:  This file contains scripts for various utility functions used in the GAN project.
    - `weights_init.py`: This file contains scripts for weight initialization functions for the GAN architecture.

- `train.py`: Script for training the model without TensorBoard logging.
- `call_methods.py`: This file contains scripts for dynamically creating models, networks, datasets, and data loaders based on provided names and options.

### Requirements

To run the code, you need the following:

- Python 3.8 or above
- PyTorch 1.7 or above
- torchvision
- tqdm
- matplotlib
- TensorboardX 2.7.0

Install the necessary packages using pip:


### Dataset

The training scripts are set up to use the Biomaterial dataset with 2176 Samples, which are loaded from the local machine. If you wish to use a different dataset, you'll need to modify the `images_utils.py` file and potentially the training scripts to handle your dataset's loading and processing.

