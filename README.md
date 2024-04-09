



### Project Structure

- `models/`: Contains the individual modules used to build the diffusion model.
    - `attention_block.py`: Defines the attention mechanisms.
    - `diffusion_model.py`: The main diffusion model class.
    - `downsampling_block.py`: Modules for downsampling feature maps.
    - `nin_block.py`: Network in network block.
    - `resnet_block.py`: ResNet blocks.
    - `timestep_embedding.py`: Embedding layers for timesteps.
    - `unet.py`: U-Net model architecture.
    - `upsampling_block.py`: Modules for upsampling feature maps.
- `options/`:
    - `base_options.py`: Command-line arguments for the training script.
- `utils/`:
    - `images_utils.py`: Utilities for image handling.
- `train.py`: Script for training the model without TensorBoard logging.
- `updated_train.py`: Script for training the model with TensorBoard logging.

### Requirements

