# Generative Adversarial Networks-for-Biomaterial-Discovery


**GANs for Image Generation using Biomaterial Topography Dataset**


This project contains the code to train a suite of GAN Variants for Biomaterial Discovery. It includes a PyTorch implementation of the U-Net model, several building blocks used in the model architecture, and scripts for training and logging.



### Project Structure


- `options/`:
    - `base_options.py`: Basic Command-line arguments for the training script.
    - `train_options.py`: Hyperparameter Command-line arguments for the training script.

- `utils/`:
    - `images_utils.py`: Utilities for image handling.
    - `custom_layers.py`: This file contains custom layers used in the GAN architecture.
    - `losses.py`: This file contains loss functions used in the GAN training process.
    - `tb_visualizer.py`: This file provides a TensorBoard visualizer for monitoring GAN training progress.
    - `utils.py`:  This file contains various utility functions used in the GAN project.
    - `weights_init.py`: This file contains weight initialization functions for the GAN architecture.

- `train.py`: Script for training the model without TensorBoard logging.
- `call_methods.py`: Script for training the model with TensorBoard logging.

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

